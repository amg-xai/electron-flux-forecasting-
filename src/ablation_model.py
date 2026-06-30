import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Flux-free feature set (no autoregression at ANY horizon) ───────────────────
ABLATION_FEATURES = [
    'B_mag', 'By_GSM', 'Bz_GSM', 'density', 'Vsw', 'Kp', 'Dst',
    'Vsw_mean_6h', 'Vsw_mean_12h', 'Vsw_mean_24h', 'Vsw_mean_48h', 'Vsw_mean_72h',
    'Bz_min_6h', 'Bz_min_12h', 'Bz_min_24h', 'Bz_min_48h', 'Bz_min_72h',
    'Bz_mean_6h', 'Bz_mean_12h', 'Bz_mean_24h', 'Bz_mean_48h', 'Bz_mean_72h',
    'density_mean_6h', 'density_mean_12h', 'density_mean_24h', 'density_mean_48h',
    'Bz_south', 'Bz_south_sum_6h', 'Bz_south_sum_12h',
    'Bz_south_sum_24h', 'Bz_south_sum_48h',
    'Pdyn', 'Ey', 'Ey_rect',
    'Ey_rect_sum_6h', 'Ey_rect_sum_12h', 'Ey_rect_sum_24h', 'Ey_rect_sum_48h',
    'hour_of_day', 'day_of_year', 'month',
]
# NOTE: No log_flux_lag features at all, no flux_trend features.


class FluxDataset(Dataset):
    def __init__(self, X, y_1h, y_6h, y_12h, seq_len=72):
        self.X    = torch.tensor(X,    dtype=torch.float32)
        self.y_1h  = torch.tensor(y_1h,  dtype=torch.float32)
        self.y_6h  = torch.tensor(y_6h,  dtype=torch.float32)
        self.y_12h = torch.tensor(y_12h, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        i = idx + self.seq_len
        return self.X[idx:i], self.y_1h[i], self.y_6h[i], self.y_12h[i]


class AblationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.head_1h  = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.head_6h  = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.head_12h = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return (
            self.head_1h(out).squeeze(-1),
            self.head_6h(out).squeeze(-1),
            self.head_12h(out).squeeze(-1),
        )


def train_ablation(features_path, model_dir, seq_len=72, epochs=50, batch_size=512, lr=1e-4):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(features_path, index_col='time', parse_dates=True)
    print(f"Loaded {len(df)} rows | Ablation features: {len(ABLATION_FEATURES)} (NO flux lags)")

    X     = df[ABLATION_FEATURES].values
    y_1h  = df['target_1h'].values
    y_6h  = df['target_6h'].values
    y_12h = df['target_12h'].values

    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[:n_train])
    X_val   = scaler.transform(X[n_train:n_val])
    X_test  = scaler.transform(X[n_val:])

    with open(os.path.join(model_dir, 'ablation_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    train_ds = FluxDataset(X_train, y_1h[:n_train], y_6h[:n_train], y_12h[:n_train], seq_len)
    val_ds   = FluxDataset(X_val,   y_1h[n_train:n_val], y_6h[n_train:n_val], y_12h[n_train:n_val], seq_len)
    test_ds  = FluxDataset(X_test,  y_1h[n_val:], y_6h[n_val:], y_12h[n_val:], seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = AblationLSTM(input_size=len(ABLATION_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss()

    best_val_loss  = float('inf')
    patience       = 10
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y1, y6, y12 in train_dl:
            X_b, y1, y6, y12 = X_b.to(device), y1.to(device), y6.to(device), y12.to(device)
            optimizer.zero_grad()
            p1, p6, p12 = model(X_b)
            loss = criterion(p1, y1) + criterion(p6, y6) + criterion(p12, y12)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y1, y6, y12 in val_dl:
                X_b, y1, y6, y12 = X_b.to(device), y1.to(device), y6.to(device), y12.to(device)
                p1, p6, p12 = model(X_b)
                val_loss += (criterion(p1, y1) + criterion(p6, y6) + criterion(p12, y12)).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'ablation_best.pt'))
            print(f"  → Saved best ablation model")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Test Evaluation ────────────────────────────────────────────────────────
    print("\n── ABLATION TEST EVALUATION (NO FLUX LAGS) ──")
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'ablation_best.pt'), map_location=device, weights_only=True
    ))
    model.eval()

    p1_all, p6_all, p12_all = [], [], []
    y1_all, y6_all, y12_all = [], [], []

    with torch.no_grad():
        for X_b, y1, y6, y12 in test_dl:
            X_b = X_b.to(device)
            p1, p6, p12 = model(X_b)
            p1_all.extend(p1.cpu().numpy())
            p6_all.extend(p6.cpu().numpy())
            p12_all.extend(p12.cpu().numpy())
            y1_all.extend(y1.numpy())
            y6_all.extend(y6.numpy())
            y12_all.extend(y12.numpy())

    results = {}
    for name, preds, trues in [('1h', p1_all, y1_all), ('6h', p6_all, y6_all), ('12h', p12_all, y12_all)]:
        preds, trues = np.array(preds), np.array(trues)
        rmse       = np.sqrt(mean_squared_error(trues, preds))
        mae        = mean_absolute_error(trues, preds)
        clim       = np.full(len(trues), np.mean(trues))
        rmse_clim  = np.sqrt(mean_squared_error(trues, clim))
        skill_clim = 1 - (rmse / rmse_clim)
        results[name] = {'rmse': rmse, 'mae': mae, 'skill_clim': skill_clim}
        print(f"  {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | Skill_clim={skill_clim:.3f}")

    return results


if __name__ == "__main__":
    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base, "data", "features.csv")
    model_dir     = os.path.join(base, "models")
    train_ablation(features_path, model_dir)
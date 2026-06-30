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
from src.features import get_feature_sets


# ── 1. DATASET ────────────────────────────────────────────────────────────────
class FluxDataset(Dataset):
    def __init__(self, X_1h, X_6h, X_12h, y_1h, y_6h, y_12h, seq_len=72):
        self.X_1h  = torch.tensor(X_1h,  dtype=torch.float32)
        self.X_6h  = torch.tensor(X_6h,  dtype=torch.float32)
        self.X_12h = torch.tensor(X_12h, dtype=torch.float32)
        self.y_1h  = torch.tensor(y_1h,  dtype=torch.float32)
        self.y_6h  = torch.tensor(y_6h,  dtype=torch.float32)
        self.y_12h = torch.tensor(y_12h, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X_1h) - self.seq_len

    def __getitem__(self, idx):
        i = idx + self.seq_len
        return (
            self.X_1h[idx:i],
            self.X_6h[idx:i],
            self.X_12h[idx:i],
            self.y_1h[i],
            self.y_6h[i],
            self.y_12h[i],
        )


# ── 2. HORIZON-SPECIFIC ENCODER ────────────────────────────────────────────────
class HorizonEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.dropout(out)


# ── 3. MULTI-HORIZON MODEL ─────────────────────────────────────────────────────
class ElectronFluxLSTM(nn.Module):
    def __init__(self, size_1h, size_6h, size_12h, hidden_size=128):
        super().__init__()
        self.encoder_1h  = HorizonEncoder(size_1h,  hidden_size)
        self.encoder_6h  = HorizonEncoder(size_6h,  hidden_size)
        self.encoder_12h = HorizonEncoder(size_12h, hidden_size)

        self.head_1h = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head_6h = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head_12h = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_1h, x_6h, x_12h):
        h1  = self.encoder_1h(x_1h)
        h6  = self.encoder_6h(x_6h)
        h12 = self.encoder_12h(x_12h)
        return (
            self.head_1h(h1).squeeze(-1),
            self.head_6h(h6).squeeze(-1),
            self.head_12h(h12).squeeze(-1),
        )


# ── 4. TRAIN ──────────────────────────────────────────────────────────────────
def train(features_path, model_dir, seq_len=72, epochs=60, batch_size=512, lr=1e-4):
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(features_path, index_col='time', parse_dates=True)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    features_1h, features_6h, features_12h = get_feature_sets()

    # Verify all features exist
    for f in features_1h + features_6h + features_12h:
        if f not in df.columns:
            raise ValueError(f"Missing feature: {f}")

    X_1h  = df[features_1h].values
    X_6h  = df[features_6h].values
    X_12h = df[features_12h].values
    y_1h  = df['target_1h'].values
    y_6h  = df['target_6h'].values
    y_12h = df['target_12h'].values

    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)

    # Separate scalers per feature set
    scaler_1h  = StandardScaler()
    scaler_6h  = StandardScaler()
    scaler_12h = StandardScaler()

    X_1h_train  = scaler_1h.fit_transform(X_1h[:n_train])
    X_6h_train  = scaler_6h.fit_transform(X_6h[:n_train])
    X_12h_train = scaler_12h.fit_transform(X_12h[:n_train])

    X_1h_val  = scaler_1h.transform(X_1h[n_train:n_val])
    X_6h_val  = scaler_6h.transform(X_6h[n_train:n_val])
    X_12h_val = scaler_12h.transform(X_12h[n_train:n_val])

    X_1h_test  = scaler_1h.transform(X_1h[n_val:])
    X_6h_test  = scaler_6h.transform(X_6h[n_val:])
    X_12h_test = scaler_12h.transform(X_12h[n_val:])

    # Save scalers and feature lists
    for name, obj in [
        ('scaler_1h', scaler_1h), ('scaler_6h', scaler_6h), ('scaler_12h', scaler_12h),
        ('features_1h', features_1h), ('features_6h', features_6h), ('features_12h', features_12h),
    ]:
        with open(os.path.join(model_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    # Datasets
    train_ds = FluxDataset(
        X_1h_train, X_6h_train, X_12h_train,
        y_1h[:n_train], y_6h[:n_train], y_12h[:n_train], seq_len
    )
    val_ds = FluxDataset(
        X_1h_val, X_6h_val, X_12h_val,
        y_1h[n_train:n_val], y_6h[n_train:n_val], y_12h[n_train:n_val], seq_len
    )
    test_ds = FluxDataset(
        X_1h_test, X_6h_test, X_12h_test,
        y_1h[n_val:], y_6h[n_val:], y_12h[n_val:], seq_len
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = ElectronFluxLSTM(
        size_1h=len(features_1h),
        size_6h=len(features_6h),
        size_12h=len(features_12h),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss()

    best_val_loss  = float('inf')
    patience       = 10
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x1, x6, x12, y1, y6, y12 in train_dl:
            x1, x6, x12 = x1.to(device), x6.to(device), x12.to(device)
            y1, y6, y12 = y1.to(device), y6.to(device), y12.to(device)
            optimizer.zero_grad()
            p1, p6, p12 = model(x1, x6, x12)
            loss = criterion(p1, y1) + criterion(p6, y6) + criterion(p12, y12)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x1, x6, x12, y1, y6, y12 in val_dl:
                x1, x6, x12 = x1.to(device), x6.to(device), x12.to(device)
                y1, y6, y12 = y1.to(device), y6.to(device), y12.to(device)
                p1, p6, p12 = model(x1, x6, x12)
                val_loss += (criterion(p1, y1) + criterion(p6, y6) + criterion(p12, y12)).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
            print(f"  → Saved best model (val={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Test Evaluation ────────────────────────────────────────────────────────
    print("\n── TEST EVALUATION ──")
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'best_model.pt'),
        map_location=device, weights_only=True
    ))
    model.eval()

    p1_all, p6_all, p12_all = [], [], []
    y1_all, y6_all, y12_all = [], [], []

    with torch.no_grad():
        for x1, x6, x12, y1, y6, y12 in test_dl:
            x1, x6, x12 = x1.to(device), x6.to(device), x12.to(device)
            p1, p6, p12  = model(x1, x6, x12)
            p1_all.extend(p1.cpu().numpy())
            p6_all.extend(p6.cpu().numpy())
            p12_all.extend(p12.cpu().numpy())
            y1_all.extend(y1.numpy())
            y6_all.extend(y6.numpy())
            y12_all.extend(y12.numpy())

    y1_all  = np.array(y1_all)
    y6_all  = np.array(y6_all)
    y12_all = np.array(y12_all)
    p1_all  = np.array(p1_all)
    p6_all  = np.array(p6_all)
    p12_all = np.array(p12_all)

    for name, preds, trues in [
        ('1h',  p1_all,  y1_all),
        ('6h',  p6_all,  y6_all),
        ('12h', p12_all, y12_all),
    ]:
        n             = len(preds)
        rmse          = np.sqrt(mean_squared_error(trues, preds))
        mae           = mean_absolute_error(trues, preds)
        clim          = np.full(n, np.mean(trues))
        rmse_clim     = np.sqrt(mean_squared_error(trues, clim))
        skill_clim    = 1 - (rmse / rmse_clim)
        # Persistence baseline
        persist       = y1_all[:n]
        rmse_persist  = np.sqrt(mean_squared_error(trues, persist))
        skill_persist = 1 - (rmse / rmse_persist) if rmse_persist > 0 else 0
        print(f"  {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | "
              f"Skill_clim={skill_clim:.3f} | Skill_persist={skill_persist:.3f}")

    print(f"\nModel saved to {model_dir}")
    return model


# ── 5. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base, "data", "features.csv")
    model_dir     = os.path.join(base, "models")
    train(features_path, model_dir)
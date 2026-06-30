import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle

ALERT_THRESHOLD = 3.5  # log10 flux — same as storm_eval.py

# ── Features: solar wind ONLY, no flux lags ────────────────────────────────────
CLASSIFIER_FEATURES = [
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


# ── Dataset ────────────────────────────────────────────────────────────────────
class StormDataset(Dataset):
    def __init__(self, X, y_6h, y_12h, seq_len=72):
        self.X    = torch.tensor(X,    dtype=torch.float32)
        self.y_6h  = torch.tensor(y_6h,  dtype=torch.float32)
        self.y_12h = torch.tensor(y_12h, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        i = idx + self.seq_len
        return self.X[idx:i], self.y_6h[i], self.y_12h[i]


# ── Model ──────────────────────────────────────────────────────────────────────
class StormClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.head_6h = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.head_12h = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return (
            torch.sigmoid(self.head_6h(out)).squeeze(-1),
            torch.sigmoid(self.head_12h(out)).squeeze(-1),
        )


# ── Train ──────────────────────────────────────────────────────────────────────
def train_classifier(features_path, model_dir, seq_len=72, epochs=50, batch_size=512, lr=1e-4):
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(features_path, index_col='time', parse_dates=True)
    print(f"Loaded {len(df)} rows")

    # Binary labels: will flux exceed threshold in next 6h / 12h?
    df['storm_6h']  = (df['target_6h']  >= ALERT_THRESHOLD).astype(int)
    df['storm_12h'] = (df['target_12h'] >= ALERT_THRESHOLD).astype(int)

    print(f"Positive rate — 6h: {df['storm_6h'].mean():.3f} | 12h: {df['storm_12h'].mean():.3f}")

    X     = df[CLASSIFIER_FEATURES].values
    y_6h  = df['storm_6h'].values
    y_12h = df['storm_12h'].values

    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)

    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X[:n_train])
    X_val    = scaler.transform(X[n_train:n_val])
    X_test   = scaler.transform(X[n_val:])

    with open(os.path.join(model_dir, 'classifier_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, 'classifier_features.pkl'), 'wb') as f:
        pickle.dump(CLASSIFIER_FEATURES, f)

    train_ds = StormDataset(X_train, y_6h[:n_train], y_12h[:n_train], seq_len)
    val_ds   = StormDataset(X_val,   y_6h[n_train:n_val], y_12h[n_train:n_val], seq_len)
    test_ds  = StormDataset(X_test,  y_6h[n_val:], y_12h[n_val:], seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = StormClassifier(input_size=len(CLASSIFIER_FEATURES)).to(device)

    # Class imbalance handling — storms are rare
    pos_rate_6h  = y_6h[:n_train].mean()
    pos_rate_12h = y_12h[:n_train].mean()
    pos_weight_6h  = torch.tensor((1 - pos_rate_6h)  / max(pos_rate_6h, 0.01)).to(device)
    pos_weight_12h = torch.tensor((1 - pos_rate_12h) / max(pos_rate_12h, 0.01)).to(device)

    criterion_6h  = nn.BCELoss()
    criterion_12h = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss  = float('inf')
    patience       = 8
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y6, y12 in train_dl:
            X_b, y6, y12 = X_b.to(device), y6.to(device), y12.to(device)
            optimizer.zero_grad()
            p6, p12 = model(X_b)

            # Weighted BCE manually for imbalance
            w6  = torch.where(y6 == 1, pos_weight_6h, torch.tensor(1.0).to(device))
            w12 = torch.where(y12 == 1, pos_weight_12h, torch.tensor(1.0).to(device))
            loss6  = (nn.functional.binary_cross_entropy(p6,  y6,  reduction='none') * w6).mean()
            loss12 = (nn.functional.binary_cross_entropy(p12, y12, reduction='none') * w12).mean()
            loss = loss6 + loss12

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y6, y12 in val_dl:
                X_b, y6, y12 = X_b.to(device), y6.to(device), y12.to(device)
                p6, p12 = model(X_b)
                loss = criterion_6h(p6, y6) + criterion_12h(p12, y12)
                val_loss += loss.item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'classifier_best.pt'))
            print(f"  → Saved best classifier")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Test Evaluation ────────────────────────────────────────────────────────
    print("\n── TEST EVALUATION ──")
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'classifier_best.pt'),
        map_location=device, weights_only=True
    ))
    model.eval()

    p6_all, p12_all = [], []
    y6_all, y12_all = [], []

    with torch.no_grad():
        for X_b, y6, y12 in test_dl:
            X_b = X_b.to(device)
            p6, p12 = model(X_b)
            p6_all.extend(p6.cpu().numpy())
            p12_all.extend(p12.cpu().numpy())
            y6_all.extend(y6.numpy())
            y12_all.extend(y12.numpy())

    for name, probs, trues in [('6h', p6_all, y6_all), ('12h', p12_all, y12_all)]:
        probs = np.array(probs)
        trues = np.array(trues)
        preds = (probs >= 0.5).astype(int)

        precision = precision_score(trues, preds, zero_division=0)
        recall    = recall_score(trues, preds, zero_division=0)
        f1        = f1_score(trues, preds, zero_division=0)
        try:
            auc = roc_auc_score(trues, probs)
        except:
            auc = float('nan')

        print(f"\n{name} Storm Classifier:")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(trues, preds)}")

    print(f"\nClassifier saved to {model_dir}")
    return model


if __name__ == "__main__":
    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base, "data", "features.csv")
    model_dir     = os.path.join(base, "models")
    train_classifier(features_path, model_dir)
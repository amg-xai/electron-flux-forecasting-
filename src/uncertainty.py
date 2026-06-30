import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ElectronFluxLSTM
from src.features import get_feature_sets

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")


def enable_mc_dropout(model):
    """Keep dropout layers active during inference, everything else in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def predict_with_uncertainty(model, x1, x6, x12, n_samples=50):
    """Run n forward passes with dropout active, return mean + std per horizon."""
    preds_1h  = []
    preds_6h  = []
    preds_12h = []

    with torch.no_grad():
        for _ in range(n_samples):
            p1, p6, p12 = model(x1, x6, x12)
            preds_1h.append(p1.item())
            preds_6h.append(p6.item())
            preds_12h.append(p12.item())

    result = {}
    for name, preds in [('1h', preds_1h), ('6h', preds_6h), ('12h', preds_12h)]:
        preds = np.array(preds)
        result[name] = {
            'mean':  float(preds.mean()),
            'std':   float(preds.std()),
            'lower': float(np.percentile(preds, 5)),
            'upper': float(np.percentile(preds, 95)),
        }
    return result


if __name__ == "__main__":
    # Quick test on a single point to verify it works
    features_1h, features_6h, features_12h = get_feature_sets()
    scalers, features = {}, {}
    for h, fl in zip(['1h','6h','12h'], [features_1h, features_6h, features_12h]):
        with open(os.path.join(MODEL_DIR, f'scaler_{h}.pkl'), 'rb') as f:
            scalers[h] = pickle.load(f)
        features[h] = fl

    model = ElectronFluxLSTM(
        size_1h=len(features_1h), size_6h=len(features_6h), size_12h=len(features_12h)
    )
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'best_model.pt'), map_location='cpu', weights_only=True
    ))
    model = enable_mc_dropout(model)

    df = pd.read_csv(os.path.join(BASE, 'data', 'features.csv'), index_col='time', parse_dates=True)

    seq_len = 72
    idx = len(df) - 200  # pick a point with enough history

    x1  = torch.tensor(scalers['1h'].transform(df[features['1h']].iloc[idx-seq_len:idx].values), dtype=torch.float32).unsqueeze(0)
    x6  = torch.tensor(scalers['6h'].transform(df[features['6h']].iloc[idx-seq_len:idx].values), dtype=torch.float32).unsqueeze(0)
    x12 = torch.tensor(scalers['12h'].transform(df[features['12h']].iloc[idx-seq_len:idx].values), dtype=torch.float32).unsqueeze(0)

    result = predict_with_uncertainty(model, x1, x6, x12, n_samples=50)

    print(f"Timestamp: {df.index[idx]}")
    print(f"Actual log_flux at this point: {df['log_flux'].iloc[idx]:.3f}")
    print()
    for horizon, stats in result.items():
        print(f"{horizon}: mean={stats['mean']:.3f} | std={stats['std']:.3f} | "
              f"90% CI=[{stats['lower']:.3f}, {stats['upper']:.3f}]")
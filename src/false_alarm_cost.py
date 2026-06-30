import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace('\\src', ''))
from src.classifier import StormClassifier, CLASSIFIER_FEATURES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

# Load classifier
with open(os.path.join(MODEL_DIR, 'classifier_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
model = StormClassifier(input_size=len(CLASSIFIER_FEATURES))
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, 'classifier_best.pt'), map_location='cpu', weights_only=True
))
model.eval()

df = pd.read_csv(os.path.join(BASE, 'data', 'features.csv'), index_col='time', parse_dates=True)

# Use test set only (last 15%)
n = len(df)
test_start = int(n * 0.85)
test_df = df.iloc[test_start:].copy()

seq_len = 72
X = scaler.transform(test_df[CLASSIFIER_FEATURES].values)
t = torch.tensor(X, dtype=torch.float32)

probs_6h = np.full(len(test_df), np.nan)
with torch.no_grad():
    for i in range(seq_len, len(test_df)):
        x = t[i-seq_len:i].unsqueeze(0)
        p6, p12 = model(x)
        probs_6h[i] = float(p6[0])

test_df['prob_6h'] = probs_6h
test_df['alert'] = test_df['prob_6h'] >= 0.5
test_df['actual_storm'] = test_df['target_6h'] >= 3.5

test_df = test_df.dropna(subset=['prob_6h'])

# Group consecutive True values into discrete episodes
def count_episodes(series):
    diffs = series.astype(int).diff().fillna(0)
    starts = (diffs == 1).sum()
    if series.iloc[0]:
        starts += 1
    return starts

alert_episodes = count_episodes(test_df['alert'])
storm_episodes = count_episodes(test_df['actual_storm'])

# True positive episodes: alert episodes that overlap with an actual storm episode
test_df['hit'] = test_df['alert'] & test_df['actual_storm']
hit_episodes = count_episodes(test_df['hit'])

false_alarm_episodes = alert_episodes - hit_episodes
missed_episodes = storm_episodes - hit_episodes

test_years = len(test_df) / (24 * 365)

print(f"Test period: {test_years:.2f} years ({len(test_df)} hours)")
print(f"Total alert episodes issued: {alert_episodes}")
print(f"Total actual storm episodes: {storm_episodes}")
print(f"Episodes where alert overlapped a real storm: {hit_episodes}")
print(f"False alarm episodes: {false_alarm_episodes}")
print(f"Missed storm episodes: {missed_episodes}")
print()
print(f"False alarms per year: {false_alarm_episodes/test_years:.1f}")
print(f"Missed storms per year: {missed_episodes/test_years:.1f}")
print(f"Detected storms per year: {hit_episodes/test_years:.1f}")
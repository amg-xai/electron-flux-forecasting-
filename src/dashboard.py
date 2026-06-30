import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ElectronFluxLSTM
from src.classifier import StormClassifier, CLASSIFIER_FEATURES
from src.features import get_feature_sets

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "features.csv")
ALERT_THRESHOLD = 3.5

st.set_page_config(page_title="Electron Flux Forecaster", page_icon="🛰️", layout="wide")


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_regression_model():
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
    model.eval()
    return model, scalers, features


@st.cache_resource
def load_classifier_model():
    with open(os.path.join(MODEL_DIR, 'classifier_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    model = StormClassifier(input_size=len(CLASSIFIER_FEATURES))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'classifier_best.pt'), map_location='cpu', weights_only=True
    ))
    model.eval()
    return model, scaler


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col='time', parse_dates=True)


def predict_regression(model, scalers, features, df, end_idx, seq_len=72):
    start_idx = end_idx - seq_len
    if start_idx < 0:
        return None
    x1  = torch.tensor(scalers['1h'].transform(df[features['1h']].iloc[start_idx:end_idx].values), dtype=torch.float32).unsqueeze(0)
    x6  = torch.tensor(scalers['6h'].transform(df[features['6h']].iloc[start_idx:end_idx].values), dtype=torch.float32).unsqueeze(0)
    x12 = torch.tensor(scalers['12h'].transform(df[features['12h']].iloc[start_idx:end_idx].values), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p1, p6, p12 = model(x1, x6, x12)
    return {'1h': float(p1[0]), '6h': float(p6[0]), '12h': float(p12[0])}


def predict_classifier(model, scaler, df, end_idx, seq_len=72):
    start_idx = end_idx - seq_len
    if start_idx < 0:
        return None
    x = torch.tensor(scaler.transform(df[CLASSIFIER_FEATURES].iloc[start_idx:end_idx].values), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p6, p12 = model(x)
    return {'6h': float(p6[0]), '12h': float(p12[0])}


def flux_to_level(log_flux):
    if log_flux >= 4.0:   return "🔴 SEVERE", "#ff4444"
    elif log_flux >= 3.5: return "🟠 HIGH", "#ff9944"
    elif log_flux >= 3.0: return "🟡 ELEVATED", "#ffdd44"
    else:                 return "🟢 NORMAL", "#44ff88"


def prob_to_risk(prob):
    if prob >= 0.7:   return "🔴 HIGH RISK", "#ff4444"
    elif prob >= 0.4: return "🟠 MODERATE RISK", "#ff9944"
    else:             return "🟢 LOW RISK", "#44ff88"


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🛰️ Electron Flux Forecasting System")
st.markdown("**ISRO Geostationary Satellite Radiation Environment Monitor — Two-Stage Pipeline**")
st.markdown("---")

reg_model, reg_scalers, reg_features = load_regression_model()
clf_model, clf_scaler = load_classifier_model()
df = load_data()

st.sidebar.header("Controls")
st.sidebar.markdown(f"**Dataset:** {df.index[0].date()} to {df.index[-1].date()}")
st.sidebar.markdown(f"**Total hours:** {len(df):,}")

max_idx = len(df) - 1
# Initialize session state for slider value
if 'slider_idx' not in st.session_state:
    st.session_state.slider_idx = max_idx - 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Jump to Storms")
storm_jumps = {
    "Oct 2015 Storm":     "2015-10-05 17:00:00",
    "Sep-Oct 2016 Storm":  "2016-09-27 16:00:00",
    "Oct 2016 Storm":      "2016-10-26 00:00:00",
    "Apr 2017 Storm":      "2017-04-21 18:00:00",
    "Oct 2017 Storm":      "2017-10-12 16:00:00",
}
for name, ts in storm_jumps.items():
    if st.sidebar.button(name):
        target_time = pd.Timestamp(ts)
        st.session_state.slider_idx = int(df.index.get_indexer([target_time], method='nearest')[0])

selected_idx = st.sidebar.slider(
    "Select time point",
    min_value=72, max_value=max_idx,
    key='slider_idx',
    step=1
)
selected_time = df.index[selected_idx]
st.sidebar.markdown(f"**Selected:** {selected_time}")
history_hours = st.sidebar.selectbox("History window", [72, 120, 168, 240], index=0)

# ── Run both models ──────────────────────────────────────────────────────────
reg_preds = predict_regression(reg_model, reg_scalers, reg_features, df, selected_idx)
clf_preds = predict_classifier(clf_model, clf_scaler, df, selected_idx)

# ── Early Warning Panel ─────────────────────────────────────────────────────────
st.subheader("⚠️ Early Warning Classifier — Storm Probability (Solar Wind Precursors Only)")
col1, col2 = st.columns(2)

if clf_preds:
    with col1:
        risk, color = prob_to_risk(clf_preds['6h'])
        st.metric("P(storm in next 6h)", f"{clf_preds['6h']*100:.1f}%")
        st.markdown(f"**{risk}**")
    with col2:
        risk, color = prob_to_risk(clf_preds['12h'])
        st.metric("P(storm in next 12h)", f"{clf_preds['12h']*100:.1f}%")
        st.markdown(f"**{risk}**")

st.caption("Classifier uses ONLY solar wind precursor features (Vsw, Bz, density, Kp, Dst) — no flux autoregression. Validated AUC: 0.90 (6h), 0.90 (12h).")

st.markdown("---")

# ── Regression Forecast Panel ────────────────────────────────────────────────────
st.subheader("📊 Flux Forecast — Multi-Horizon LSTM")
col1, col2, col3 = st.columns(3)

if reg_preds:
    with col1:
        level, color = flux_to_level(reg_preds['1h'])
        st.metric("1-Hour Forecast", f"{10**reg_preds['1h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['1h']:.2f}")
        st.markdown(f"**Status: {level}**")
    with col2:
        level, color = flux_to_level(reg_preds['6h'])
        st.metric("6-Hour Forecast", f"{10**reg_preds['6h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['6h']:.2f}")
        st.markdown(f"**Status: {level}**")
    with col3:
        level, color = flux_to_level(reg_preds['12h'])
        st.metric("12-Hour Forecast", f"{10**reg_preds['12h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['12h']:.2f}")
        st.markdown(f"**Status: {level}**")

st.markdown("---")

# ── Light Curve Plot ────────────────────────────────────────────────────────────
st.subheader("📈 Electron Flux Light Curve")

plot_start = max(0, selected_idx - history_hours)
plot_df = df.iloc[plot_start:selected_idx+1]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.patch.set_facecolor('#0e1117')

ax1 = axes[0]
ax1.set_facecolor('#0e1117')
ax1.plot(plot_df.index, plot_df['log_flux'], color='cyan', linewidth=1, label='>2 MeV flux')
ax1.axhline(y=3.5, color='orange', linestyle='--', alpha=0.7, label='HIGH threshold')
ax1.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='SEVERE threshold')
ax1.axvline(x=selected_time, color='white', linestyle=':', alpha=0.8, label='Now')

if reg_preds:
    future_times = [selected_time + timedelta(hours=1), selected_time + timedelta(hours=6), selected_time + timedelta(hours=12)]
    future_vals  = [reg_preds['1h'], reg_preds['6h'], reg_preds['12h']]
    ax1.scatter(future_times, future_vals, color='yellow', zorder=5, s=80, label='Regression forecast')
    ax1.plot([selected_time] + future_times, [plot_df['log_flux'].iloc[-1]] + future_vals, color='yellow', linestyle='--', alpha=0.6)

ax1.set_ylabel('log₁₀ Flux', color='white')
ax1.tick_params(colors='white')
ax1.legend(loc='upper left', facecolor='#1e1e1e', labelcolor='white', fontsize=8)
ax1.set_title('Electron Flux >2 MeV (GOES-15)', color='white')
for spine in ax1.spines.values(): spine.set_edgecolor('#333')

ax2 = axes[1]
ax2.set_facecolor('#0e1117')
ax2.plot(plot_df.index, plot_df['Vsw'], color='lime', linewidth=1)
ax2.axhline(y=600, color='orange', linestyle='--', alpha=0.5, label='High speed')
ax2.axvline(x=selected_time, color='white', linestyle=':', alpha=0.8)
ax2.set_ylabel('Vsw (km/s)', color='white')
ax2.tick_params(colors='white')
ax2.set_title('Solar Wind Speed', color='white')
ax2.legend(fontsize=8)
for spine in ax2.spines.values(): spine.set_edgecolor('#333')

ax3 = axes[2]
ax3.set_facecolor('#0e1117')
ax3.plot(plot_df.index, plot_df['Bz_GSM'], color='magenta', linewidth=1)
ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='Storm threshold')
ax3.fill_between(plot_df.index, plot_df['Bz_GSM'], 0, where=plot_df['Bz_GSM'] < 0, color='red', alpha=0.2, label='Southward Bz')
ax3.set_ylabel('Bz GSM (nT)', color='white')
ax3.tick_params(colors='white')
ax3.set_title('IMF Bz (Southward = Storm Driver)', color='white')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, color='white')
ax3.legend(fontsize=8)
for spine in ax3.spines.values(): spine.set_edgecolor('#333')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Storm Event Table ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Major Storm Events in Dataset")
storm_events = df[df['log_flux'] > 4.0]
if len(storm_events) > 0:
    st.markdown(f"Found **{len(storm_events)}** hours with severe flux (log₁₀ > 4.0)")
    storm_summary = storm_events[['log_flux','Vsw','Bz_GSM','Kp','Dst']].head(20).copy()
    storm_summary['flux_actual'] = 10**storm_summary['log_flux']
    st.dataframe(storm_summary, use_container_width=True)

# ── Model Performance ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Model Architecture & Validated Performance"):
    st.markdown("""
    ### Two-Stage Forecasting Pipeline

    **Stage 1 — Early Warning Classifier**
    Binary classification using ONLY solar wind precursor features (no flux history), forcing the model
    to learn genuine solar-wind-to-storm causality rather than autoregressive shortcuts.
    - Architecture: LSTM (2 layers, 96 hidden units) + dual sigmoid heads
    - Test AUC: 0.909 (6h) | 0.898 (12h)
    - Test Recall: 0.831 (6h) | 0.833 (12h)
    - Validated lead time (storm-event level, 6-hour persistence filter): 4-13 hours on 4/5 major storms (2015-2019)

    **Stage 2 — Multi-Horizon Flux Regression**
    Separate LSTM encoders per forecast horizon with horizon-appropriate feature sets
    (longer horizons exclude short-term flux lags to reduce autoregressive shortcuts).
    - Architecture: 3x independent LSTM encoders (1h/6h/12h) + separate dense heads
    - Test RMSE (log₁₀ flux): 0.223 (1h) | 0.333 (6h) | 0.378 (12h)
    - Skill vs. climatology: 0.71 (1h) | 0.57 (6h) | 0.52 (12h)

    **Known Limitation**
    The classifier underperforms on impulsive, velocity-driven storm onsets (e.g. April 2017,
    8.08 km/s/hour Vsw rise rate) compared to gradual, Bz-sustained storms (e.g. October 2015,
    cumulative southward Bz exposure 116 nT·h over 48h). This is consistent with the physical
    distinction between CIR-driven and shock-driven geomagnetic storms.

    **Training Data:** GOES-15 EPEAD >2 MeV electron flux + OMNI solar wind parameters (2015-2019, hourly resolution)
    """)
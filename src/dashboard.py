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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.model import ElectronFluxLSTM
from src.magnetosphere_viz import generate_magnetosphere_html
import streamlit.components.v1 as components
from src.classifier import StormClassifier, CLASSIFIER_FEATURES
from src.features import get_feature_sets
from src.uncertainty import enable_mc_dropout, predict_with_uncertainty

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "features.csv")
ALERT_THRESHOLD = 3.5

st.set_page_config(page_title="Electron Flux Forecaster", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
    .main { background-color: #0a0e14; }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #131a24 0%, #0d1117 100%);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px 20px;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .risk-low { background: rgba(68, 255, 136, 0.15); color: #44ff88; border: 1px solid rgba(68, 255, 136, 0.3); }
    .risk-moderate { background: rgba(255, 153, 68, 0.15); color: #ff9944; border: 1px solid rgba(255, 153, 68, 0.3); }
    .risk-high { background: rgba(255, 68, 68, 0.15); color: #ff4444; border: 1px solid rgba(255, 68, 68, 0.3); }
    
    h1 { 
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    h2, h3 { color: #e5e7eb; font-weight: 600; }
    
    .stButton button {
        background: #1a2332;
        border: 1px solid #2d3748;
        color: #e5e7eb;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: #2d3748;
        border-color: #00d4ff;
    }
    
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1f2937;
    }
    
    .stCaption { color: #6b7280; }
    
    hr { border-color: #1f2937; }
</style>
""", unsafe_allow_html=True)

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
    model = enable_mc_dropout(model)
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
    result = predict_with_uncertainty(model, x1, x6, x12, n_samples=30)
    return {
        '1h':  result['1h']['mean'],
        '6h':  result['6h']['mean'],
        '12h': result['12h']['mean'],
        '1h_ci':  (result['1h']['lower'], result['1h']['upper']),
        '6h_ci':  (result['6h']['lower'], result['6h']['upper']),
        '12h_ci': (result['12h']['lower'], result['12h']['upper']),
    }


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
st.markdown("""
<div style="padding: 20px 0 10px 0;">
    <h1 style="margin-bottom: 4px;">Electron Flux Forecasting System</h1>
    <p style="color: #6b7280; font-size: 1rem; margin-top: 0;">
        ISRO Geostationary Satellite Radiation Environment Monitor &nbsp;·&nbsp; Two-Stage Pipeline
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

reg_model, reg_scalers, reg_features = load_regression_model()
clf_model, clf_scaler = load_classifier_model()
df = load_data()

st.sidebar.header("Controls")
st.sidebar.markdown(f"**Dataset:** {df.index[0].date()} to {df.index[-1].date()}")
st.sidebar.markdown(f"**Total hours:** {len(df):,}")

max_idx = len(df) - 1
# Initialize session state for slider value.
# Default to a storm onset (not a quiet moment) so the dashboard opens on a
# visually dynamic, high-impact state rather than a flat "normal" period.
if 'slider_idx' not in st.session_state:
    default_storm = pd.Timestamp("2015-10-05 17:00:00")
    try:
        st.session_state.slider_idx = int(df.index.get_indexer([default_storm], method='nearest')[0])
    except Exception:
        st.session_state.slider_idx = max_idx - 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Jump to Storms")
storm_jumps = {
    "Mar 2015 Storm":     "2015-03-17 14:00:00",
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

# ── Magnetosphere visualization — at-a-glance physical state ──────────────────
vsw_now = float(df['Vsw'].iloc[selected_idx])
bz_now = float(df['Bz_GSM'].iloc[selected_idx])
flux_now = float(df['log_flux'].iloc[selected_idx])

st.markdown("""
<div style="background: linear-gradient(135deg, #0a0e14 0%, #0d1117 100%); 
            border: 1px solid #1f2937; border-left: 3px solid #7c3aed;
            border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
    <h3 style="margin: 0; color: #a78bfa;">Magnetosphere — live physical state</h3>
    <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">Solar wind interaction at selected timestamp</p>
</div>
""", unsafe_allow_html=True)

components.html(generate_magnetosphere_html(vsw_now, bz_now, flux_now), height=520)

st.markdown("---")

# ── Early Warning Panel ─────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1410 0%, #0d1117 100%); 
            border: 1px solid #2d2417; border-left: 3px solid #ff9944;
            border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
    <h3 style="margin: 0; color: #ff9944;">Early warning classifier — storm probability</h3>
    <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">Solar wind precursors only</p>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)

def risk_badge_html(prob):
    if prob >= 0.7:
        return '<span class="risk-badge risk-high">● HIGH RISK</span>'
    elif prob >= 0.4:
        return '<span class="risk-badge risk-moderate">● MODERATE RISK</span>'
    else:
        return '<span class="risk-badge risk-low">● LOW RISK</span>'

if clf_preds:
    with col1:
        st.metric("P(storm in next 6h)", f"{clf_preds['6h']*100:.1f}%")
        st.markdown(risk_badge_html(clf_preds['6h']), unsafe_allow_html=True)
    with col2:
        st.metric("P(storm in next 12h)", f"{clf_preds['12h']*100:.1f}%")
        st.markdown(risk_badge_html(clf_preds['12h']), unsafe_allow_html=True)

st.caption("Classifier uses ONLY solar wind precursor features (Vsw, Bz, density, Kp, Dst) — no flux autoregression. Validated AUC: 0.90 (6h), 0.90 (12h).")

st.markdown("---")

# ── Regression Forecast Panel ────────────────────────────────────────────────────
st.subheader("Flux Forecast — Multi-Horizon LSTM")
col1, col2, col3 = st.columns(3)

if reg_preds:
    with col1:
        level, color = flux_to_level(reg_preds['1h'])
        lo, hi = reg_preds['1h_ci']
        st.metric("1-Hour Forecast", f"{10**reg_preds['1h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['1h']:.2f}")
        st.markdown(f"**Status: {level}**")
        st.caption(f"90% CI: [{10**lo:.0f}, {10**hi:.0f}]")
    with col2:
        level, color = flux_to_level(reg_preds['6h'])
        lo, hi = reg_preds['6h_ci']
        st.metric("6-Hour Forecast", f"{10**reg_preds['6h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['6h']:.2f}")
        st.markdown(f"**Status: {level}**")
        st.caption(f"90% CI: [{10**lo:.0f}, {10**hi:.0f}]")
    with col3:
        level, color = flux_to_level(reg_preds['12h'])
        lo, hi = reg_preds['12h_ci']
        st.metric("12-Hour Forecast", f"{10**reg_preds['12h']:.0f} e/cm²/s/sr", f"log₁₀={reg_preds['12h']:.2f}")
        st.markdown(f"**Status: {level}**")
        st.caption(f"90% CI: [{10**lo:.0f}, {10**hi:.0f}]")

# ── Forecast vs Actual (verification) ────────────────────────────────────────
# Because forecasts run over historical data, the ground-truth flux at each
# horizon is known. Showing it here turns "trust our metrics" into "see for
# yourself" — the core honesty principle of the project.
if reg_preds:
    ver_rows = []
    for h, key in zip([1, 6, 12], ['1h', '6h', '12h']):
        fut_idx = selected_idx + h
        if fut_idx <= max_idx:
            actual_log = df['log_flux'].iloc[fut_idx]
            pred_log = reg_preds[key]
            ver_rows.append({
                'Horizon': f'{h}h',
                'Predicted (log₁₀)': f'{pred_log:.2f}',
                'Actual (log₁₀)': f'{actual_log:.2f}',
                'Error (log₁₀)': f'{abs(pred_log - actual_log):.2f}',
            })
    if ver_rows:
        with st.expander("Forecast vs actual — verification against ground truth", expanded=True):
            st.caption("Forecasts run over historical data, so we can show what actually "
                       "happened at each horizon. This is verification, not assertion.")
            st.dataframe(pd.DataFrame(ver_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Selected time is near the end of the dataset — no future ground truth available "
                "to verify against. Pick an earlier timestamp to see forecast-vs-actual.")

st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #0d1a1f 0%, #0d1117 100%); 
            border: 1px solid #1a2d33; border-left: 3px solid #00d4ff;
            border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
    <h3 style="margin: 0; color: #00d4ff;">Electron flux light curve</h3>
    <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">Hover any point for exact values · drag to zoom · click legend to toggle</p>
</div>
""", unsafe_allow_html=True)

plot_start = max(0, selected_idx - history_hours)
plot_df = df.iloc[plot_start:selected_idx+1]

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=("Electron Flux >2 MeV (GOES-15)", "Solar Wind Speed", "IMF Bz (Southward = Storm Driver)")
)

# Panel 1: Flux
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['log_flux'], mode='lines', name='>2 MeV flux',
    line=dict(color='cyan', width=1.5),
    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>log₁₀ flux: %{y:.2f}<extra></extra>'
), row=1, col=1)

fig.add_hline(y=3.5, line_dash="dash", line_color="orange", opacity=0.6, row=1, col=1)
fig.add_hline(y=4.0, line_dash="dash", line_color="red", opacity=0.6, row=1, col=1)

if reg_preds:
    horizons = [1, 6, 12]
    future_times = [selected_time + timedelta(hours=h) for h in horizons]
    future_vals  = [reg_preds['1h'], reg_preds['6h'], reg_preds['12h']]
    fig.add_trace(go.Scatter(
        x=future_times, y=future_vals, mode='markers+lines', name='Forecast',
        line=dict(color='yellow', width=2, dash='dash'),
        marker=dict(color='yellow', size=10, symbol='circle'),
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Forecast log₁₀: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Actual-vs-predicted overlay: show ground-truth flux at the forecast horizons
    # (available because we forecast over historical data). This lets a viewer verify
    # the forecast against what really happened, not just trust the numbers.
    actual_times, actual_vals = [], []
    for h in horizons:
        fut_idx = selected_idx + h
        if fut_idx <= max_idx:
            actual_times.append(df.index[fut_idx])
            actual_vals.append(df['log_flux'].iloc[fut_idx])
    if actual_vals:
        fig.add_trace(go.Scatter(
            x=actual_times, y=actual_vals, mode='markers', name='Actual (ground truth)',
            marker=dict(color='#00ff9d', size=11, symbol='x', line=dict(width=2, color='#00ff9d')),
            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Actual log₁₀: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

# Panel 2: Solar wind speed
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['Vsw'], mode='lines', name='Vsw',
    line=dict(color='lime', width=1.5),
    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Vsw: %{y:.0f} km/s<extra></extra>'
), row=2, col=1)
fig.add_hline(y=600, line_dash="dash", line_color="orange", opacity=0.4, row=2, col=1)

# Panel 3: Bz
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['Bz_GSM'], mode='lines', name='Bz',
    line=dict(color='magenta', width=1.5),
    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Bz: %{y:.1f} nT<extra></extra>'
), row=3, col=1)
fig.add_hline(y=0, line_color="gray", opacity=0.4, row=3, col=1)

# "Now" vertical line across all panels
for r in [1, 2, 3]:
    fig.add_vline(x=selected_time, line_dash="dot", line_color="white", opacity=0.6, row=r, col=1)

fig.update_layout(
    height=650, template="plotly_dark",
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=30, t=60, b=40)
)
fig.update_yaxes(title_text="log₁₀ Flux", row=1, col=1)
fig.update_yaxes(title_text="Vsw (km/s)", row=2, col=1)
fig.update_yaxes(title_text="Bz GSM (nT)", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── ISRO GRASP/GSAT-19 Validation (PS requirement) ──────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #0d1f1a 0%, #0d1117 100%);
            border: 1px solid #1a3329; border-left: 3px solid #1d9e75;
            border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
    <h3 style="margin: 0; color: #2ecc8f;">Independent Validation — ISRO GRASP / GSAT-19</h3>
    <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">
        The problem statement asks for validation against ISRO's own satellite data. We trained on
        GOES-15, then tested whether the model generalizes to GRASP/GSAT-19 at Indian longitude.
    </p>
</div>
""", unsafe_allow_html=True)
gcol1, gcol2, gcol3 = st.columns(3)
with gcol1:
    st.metric("Correlation (Pearson r)", "0.614", help="Model predictions vs actual GRASP/GSAT-19 flux")
with gcol2:
    st.metric("Sample size", "3,686 h", help="Overlapping hours, 2017–2018")
with gcol3:
    st.metric("Significance", "p < 0.001", help="Correlation is highly statistically significant")
st.caption("Trained on GOES-15 (2015–2019); tested on ISRO GRASP/GSAT-19 (2017–2018) at Indian "
           "longitude — an independent instrument the model never saw during training. A positive, "
           "significant correlation on unseen ISRO data is direct evidence of real generalization, "
           "not curve-fitting.")

# ── Elevated Flux Table ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Elevated Flux Hours in Dataset")
storm_events = df[df['log_flux'] > 4.0]
if len(storm_events) > 0:
    pct = 100 * len(storm_events) / len(df)
    st.markdown(
        f"Found **{len(storm_events)}** hours ({pct:.1f}% of the dataset) with elevated flux "
        f"(log₁₀ > 4.0). These are individual *hours* of raised flux — distinct sustained "
        f"**storm events** (clustered elevated hours driven by a single disturbance) are far "
        f"rarer, which is exactly why the early-warning classifier targets event onset rather "
        f"than hourly threshold crossings."
    )
    storm_summary = storm_events[['log_flux','Vsw','Bz_GSM','Kp','Dst']].head(20).copy()
    storm_summary['flux_actual'] = 10**storm_summary['log_flux']
    storm_summary = storm_summary.rename(columns={
        'log_flux': 'log₁₀ flux',
        'Vsw': 'Vsw (km/s)',
        'Bz_GSM': 'Bz GSM (nT)',
        'Kp': 'Kp index',
        'Dst': 'Dst (nT)',
        'flux_actual': 'Flux (e/cm²/s/sr)',
    })
    st.dataframe(storm_summary, use_container_width=True)

# ── Model Performance ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Model Architecture & Validated Performance"):
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
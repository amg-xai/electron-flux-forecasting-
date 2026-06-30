import pandas as pd
import numpy as np
import os


def engineer_features(df):
    feat = df.copy()

    # ── 1. SOLAR WIND ROLLING STATISTICS ──────────────────────────────────────
    for window in [6, 12, 24, 48, 72]:
        feat[f'Vsw_mean_{window}h']     = feat['Vsw'].rolling(window).mean()
        feat[f'Bz_min_{window}h']       = feat['Bz_GSM'].rolling(window).min()
        feat[f'Bz_mean_{window}h']      = feat['Bz_GSM'].rolling(window).mean()
        feat[f'density_mean_{window}h'] = feat['density'].rolling(window).mean()

    # ── 2. SOUTHWARD BZ EXPOSURE ───────────────────────────────────────────────
    feat['Bz_south'] = feat['Bz_GSM'].clip(upper=0).abs()
    for window in [6, 12, 24, 48]:
        feat[f'Bz_south_sum_{window}h'] = feat['Bz_south'].rolling(window).sum()

    # ── 3. SOLAR WIND DYNAMIC PRESSURE ────────────────────────────────────────
    feat['Pdyn'] = 1.67e-6 * feat['density'] * feat['Vsw'] ** 2

    # ── 4. RECTIFIED ELECTRIC FIELD ───────────────────────────────────────────
    feat['Ey']      = -feat['Vsw'] * feat['Bz_GSM'] / 1000.0
    feat['Ey_rect'] = feat['Ey'].clip(lower=0)
    for window in [6, 12, 24, 48]:
        feat[f'Ey_rect_sum_{window}h'] = feat['Ey_rect'].rolling(window).sum()

    # ── 5. FLUX LAG FEATURES ──────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        feat[f'log_flux_lag_{lag}h'] = feat['log_flux'].shift(lag)

    # ── 6. FLUX TREND ─────────────────────────────────────────────────────────
    feat['flux_trend_6h']  = feat['log_flux'] - feat['log_flux'].shift(6)
    feat['flux_trend_24h'] = feat['log_flux'] - feat['log_flux'].shift(24)

    # ── 7. TIME FEATURES ──────────────────────────────────────────────────────
    feat['hour_of_day'] = feat.index.hour
    feat['day_of_year'] = feat.index.dayofyear
    feat['month']       = feat.index.month

    # ── 8. TARGET VARIABLES ───────────────────────────────────────────────────
    feat['target_1h']  = feat['log_flux'].shift(-1)
    feat['target_6h']  = feat['log_flux'].shift(-6)
    feat['target_12h'] = feat['log_flux'].shift(-12)

    feat = feat.dropna()

    print(f"Feature matrix: {feat.shape}")
    return feat


def get_feature_sets():
    """
    Return separate feature sets per forecast horizon.
    Longer horizons use fewer autoregressive features
    to force the model to learn solar wind precursors.
    """
    solar_wind_base = [
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
        'flux_trend_6h', 'flux_trend_24h',
    ]

    # 1h: all features including short lags (autoregression valid)
    features_1h = solar_wind_base + [
        'log_flux_lag_1h', 'log_flux_lag_2h', 'log_flux_lag_3h',
        'log_flux_lag_6h', 'log_flux_lag_12h', 'log_flux_lag_24h', 'log_flux_lag_48h',
    ]

    # 6h: only lags >= 6h (can't use flux from last 6h to predict 6h ahead)
    features_6h = solar_wind_base + [
        'log_flux_lag_6h', 'log_flux_lag_12h', 'log_flux_lag_24h', 'log_flux_lag_48h',
    ]

    # 12h: only lags >= 12h (pure solar wind driven)
    features_12h = solar_wind_base + [
        'log_flux_lag_12h', 'log_flux_lag_24h', 'log_flux_lag_48h',
    ]

    return features_1h, features_6h, features_12h


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(
        os.path.join(base, "data", "merged_clean.csv"),
        index_col='time', parse_dates=True
    )
    feat = engineer_features(df)
    out  = os.path.join(base, "data", "features.csv")
    feat.to_csv(out)
    print(f"Saved to {out}")
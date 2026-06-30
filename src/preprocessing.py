import pandas as pd
import numpy as np
import os
import glob


# ── 1. LOAD OMNI SOLAR WIND DATA ──────────────────────────────────────────────
def load_omni(path):
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=['year','doy','hour','B_mag','By_GSM','Bz_GSM','density','Vsw','Kp','Dst'],
        na_values=[999.9, 9999., 99999, 9999.99, 99.99, 9999]
    )
    df = df.dropna(subset=['year','doy','hour'])
    df['time'] = pd.to_datetime(
        df['year'].astype(int).astype(str) +
        df['doy'].astype(int).astype(str).str.zfill(3) +
        df['hour'].astype(int).astype(str).str.zfill(2),
        format='%Y%j%H'
    )
    df = df[['time','B_mag','By_GSM','Bz_GSM','density','Vsw','Kp','Dst']]
    df = df.set_index('time').sort_index()
    df['Kp'] = df['Kp'] / 10.0
    print(f"OMNI loaded: {len(df)} rows | {df.index[0]} to {df.index[-1]}")
    return df


# ── 2. LOAD GOES ELECTRON FLUX DATA ───────────────────────────────────────────
def load_goes(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "g15_epead_e13ew_1m_*.csv")))
    print(f"Found {len(files)} GOES files")

    dfs = []
    for f in files:
        try:
            # Find header row dynamically
            with open(f, 'r') as fh:
                lines = fh.readlines()

            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith('time_tag,'):
                    header_idx = i
                    break

            if header_idx is None:
                print(f"No header found in {os.path.basename(f)}, skipping")
                continue

            df = pd.read_csv(f, skiprows=header_idx)

            # Clean column names
            df.columns = df.columns.str.strip()

            if 'time_tag' not in df.columns:
                print(f"No time_tag in {os.path.basename(f)}, skipping")
                continue

            df['time'] = pd.to_datetime(df['time_tag'], errors='coerce')
            df = df.dropna(subset=['time'])
            df = df.set_index('time')

            # Try all possible >2 MeV column name patterns
            flux_col_pairs = [
                ('E2E_UNCOR_FLUX', 'E2W_UNCOR_FLUX'),
                ('E2E_COR_FLUX',   'E2W_COR_FLUX'),
                ('E2E_FLUX',       'E2W_FLUX'),
            ]

            flux_found = False
            for east_col, west_col in flux_col_pairs:
                if east_col in df.columns and west_col in df.columns:
                    df['flux_2MeV'] = (
                        pd.to_numeric(df[east_col], errors='coerce') +
                        pd.to_numeric(df[west_col], errors='coerce')
                    ) / 2.0
                    flux_found = True
                    break
                elif east_col in df.columns:
                    df['flux_2MeV'] = pd.to_numeric(df[east_col], errors='coerce')
                    flux_found = True
                    break

            if not flux_found:
                print(f"No flux column in {os.path.basename(f)}")
                print(f"  Available: {[c for c in df.columns if 'FLUX' in c.upper()]}")
                continue

            df = df[['flux_2MeV']].copy()
            df = df[df['flux_2MeV'] > 0]
            df = df[df['flux_2MeV'] < 1e6]
            df = df.dropna()

            if len(df) > 0:
                dfs.append(df)
                print(f"  Loaded {os.path.basename(f)}: {len(df)} rows")

        except Exception as e:
            print(f"Error loading {os.path.basename(f)}: {e}")

    if not dfs:
        raise ValueError("No GOES files loaded successfully. Check data directory.")

    goes = pd.concat(dfs).sort_index()
    # Remove duplicate timestamps
    goes = goes[~goes.index.duplicated(keep='first')]
    goes['log_flux'] = np.log10(goes['flux_2MeV'])
    print(f"\nGOES loaded: {len(goes)} rows | {goes.index[0]} to {goes.index[-1]}")
    return goes


# ── 3. MERGE AND RESAMPLE TO HOURLY ───────────────────────────────────────────
def merge_datasets(omni, goes):
    # Resample GOES from 1-min to hourly
    goes_hourly = goes.resample('1h').mean()

    # Merge on timestamp
    merged = omni.join(goes_hourly, how='inner')

    # Drop rows missing key variables
    merged = merged.dropna(subset=['Vsw', 'Bz_GSM', 'log_flux'])

    # Remove any remaining infinities
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\nMerged dataset: {len(merged)} rows")
    print(f"Date range: {merged.index[0]} to {merged.index[-1]}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nBasic stats:\n{merged.describe().round(3)}")
    return merged


# ── 4. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    omni_file = os.path.join(base, "data", "omni_data", "omni2_qGLlOaGRiE.lst")
    goes_dir  = os.path.join(base, "data", "goes_data")

    omni   = load_omni(omni_file)
    goes   = load_goes(goes_dir)
    merged = merge_datasets(omni, goes)

    out = os.path.join(base, "data", "merged_clean.csv")
    merged.to_csv(out)
    print(f"\nSaved to {out}")
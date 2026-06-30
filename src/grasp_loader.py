import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime, timedelta


def parse_grasp_file(filepath):
    """Parse a single GRASP daily txt file."""
    df = pd.read_csv(
        filepath, sep=r'\s+', skiprows=1,
        names=['doy_frac', 'electron_flux', 'proton_flux']
    )

    # Extract year from filename: grasp_5_min_avg_9-JAN-2018.txt
    fname = os.path.basename(filepath)
    match = re.search(r'(\d{1,2})-(\w{3})-(\d{4})', fname)
    if not match:
        return None
    year = int(match.group(3))

    # Convert day-of-year fractional to datetime
    base = datetime(year, 1, 1)
    df['time'] = df['doy_frac'].apply(lambda d: base + timedelta(days=d - 1))

    df = df[['time', 'electron_flux', 'proton_flux']].set_index('time')
    return df


def load_all_grasp(grasp_dir):
    files = sorted(glob.glob(os.path.join(grasp_dir, "*.txt")))
    print(f"Found {len(files)} GRASP files")

    dfs = []
    for f in files:
        try:
            df = parse_grasp_file(f)
            if df is not None and len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error parsing {os.path.basename(f)}: {e}")

    if not dfs:
        raise ValueError("No GRASP files loaded")

    grasp = pd.concat(dfs).sort_index()
    grasp = grasp[~grasp.index.duplicated(keep='first')]

    # Remove invalid/negative values
    grasp = grasp[grasp['electron_flux'] > 0]

    print(f"GRASP loaded: {len(grasp)} rows | {grasp.index[0]} to {grasp.index[-1]}")
    return grasp


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grasp_dir = os.path.join(base, "data", "grasp_data", "extracted")

    grasp = load_all_grasp(grasp_dir)

    out = os.path.join(base, "data", "grasp_clean.csv")
    grasp.to_csv(out)
    print(f"Saved to {out}")

    print("\nSample data:")
    print(grasp.head(10))
    print("\nStats:")
    print(grasp.describe())
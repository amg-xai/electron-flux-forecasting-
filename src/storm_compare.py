import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base, 'data', 'features.csv'), index_col='time', parse_dates=True)

storms = {
    'Oct 2015 (worked +8h)':      ('2015-10-03', '2015-10-05 17:00:00'),
    'Sep-Oct 2016 (worked +4h)':  ('2016-09-25', '2016-09-27 16:00:00'),
    'Apr 2017 (FAILED -12h)':     ('2017-04-19', '2017-04-21 18:00:00'),
    'Oct 2017 (worked +6h)':      ('2017-10-10', '2017-10-12 16:00:00'),
}

print(f"{'Storm':<28} {'Vsw_rise':>10} {'Bz_min':>8} {'BzS_sum48h':>11} {'Pdyn_max':>10} {'onset_speed':>12}")
print("-" * 85)

for name, (start, onset) in storms.items():
    window = df[start:onset]

    vsw_start = window['Vsw'].iloc[0]
    vsw_end   = window['Vsw'].iloc[-1]
    vsw_rise  = vsw_end - vsw_start

    bz_min       = window['Bz_GSM'].min()
    bz_south_sum = window['Bz_south_sum_48h'].iloc[-1]
    pdyn_max     = window['Pdyn'].max()

    last_12h    = window['Vsw'].iloc[-12:]
    onset_speed = (last_12h.iloc[-1] - last_12h.iloc[0]) / 12

    print(f"{name:<28} {vsw_rise:>10.1f} {bz_min:>8.2f} {bz_south_sum:>11.1f} {pdyn_max:>10.2f} {onset_speed:>12.2f}")
# Electron Flux Forecasting System
### ISRO Geostationary Satellite Radiation Environment Monitor

A two-stage machine learning pipeline that predicts >2 MeV electron flux at geostationary orbit, providing 30-45 minute to 12-hour advance warning for radiation events that threaten ISRO's GSAT satellite fleet.

Built for Bharat Antriksh Hackathon — Problem Statement: *Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites*

---

## The problem

Energetic electrons trapped in Earth's outer radiation belt cause deep dielectric charging in geostationary satellites — a leading cause of in-orbit electronics failure. Predicting flux spikes before they occur gives satellite operators time to put payloads into safe mode.

## Our approach

Rather than a single model, we built two purpose-specific systems that work together:

**1. Early warning classifier** — predicts probability of a storm in the next 6h/12h using *only* solar wind precursor data (speed, IMF Bz, density, Kp, Dst). No flux history is given to this model, forcing it to learn genuine causal solar-wind-to-storm relationships rather than taking the easy autoregressive shortcut.

**2. Multi-horizon flux regression** — predicts the actual flux value at 1h, 6h, and 12h ahead, with separate LSTM encoders per horizon using horizon-appropriate feature sets.

## Key results

| Component | Metric | Result |
|---|---|---|
| Early warning classifier | AUC (6h / 12h) | 0.909 / 0.898 |
| Early warning classifier | Recall (6h / 12h) | 0.831 / 0.833 |
| Early warning classifier | Storm lead time (5 major storms, persistence-filtered) | 4/5 storms: +4 to +13 hours |
| Flux regression | RMSE log₁₀ flux (1h / 6h / 12h) | 0.223 / 0.333 / 0.378 |
| Flux regression | Skill vs climatology (1h / 6h / 12h) | 0.71 / 0.57 / 0.52 |
| Independent validation | GOES-15 vs GRASP/GSAT-19 correlation | r = 0.604, p < 0.001, n = 3,739h |

## Why two models

Our first attempt was a single LSTM predicting flux directly. It achieved strong RMSE but **negative lead time on 4/5 storms** — it was detecting storms after they started, not before, because the model exploited flux autoregression as a shortcut.

We confirmed this with an ablation study: removing all flux-lag features costs 101% RMSE at the 1-hour horizon but only 29% at 12 hours — proof that longer-horizon predictions already depend more heavily on genuine solar wind physics than on recent flux history.

The fix was architectural: a dedicated classifier trained with zero flux history, forced to learn precursor signatures. This single change took average lead time from negative to +4 to +13 hours on 4 of 5 major storms.

## Known limitation

The classifier failed to give early warning on the April 2017 storm. Investigation showed this storm had an unusually fast, impulsive solar wind onset (8.08 km/s/hour rise rate vs 0.3-4 km/s/hour for successful detections) with weak sustained southward Bz exposure (33.5 nT·h vs 88-118 nT·h for storms we caught early). This is consistent with the known physical distinction between gradual CIR-driven storms and sudden shock-driven storms — our model has learned the former pattern well but has reduced sensitivity to the latter.

## Validation against ISRO data

The PS specifically asks for validation against ISRO's own GRASP/GSAT-19 payload. We retrieved 410 days of GRASP daily files from the PRADAN portal (185 successfully parsed after filtering corrupted downloads), covering July 2017 - January 2018. Despite GRASP and GOES-15 using different instruments, different energy thresholds, and being positioned at opposite longitudes, electron flux measurements correlate at r=0.604 (p<0.001) — strong evidence that both instruments are tracking the same underlying radiation belt physics, and that our model's findings generalize to the Indian operational environment.

## Architecture

See `architecture_diagram.png` for the full pipeline.

GOES-15 + OMNI/Wind + GRASP  →  Feature engineering  →  Early warning classifier (solar wind only)
→  Multi-horizon LSTM regression (flux + solar wind)
→  Storm evaluation + GRASP cross-validation
→  Interactive dashboard

## Project structure
electron_flux_project/
├── data/                       # GOES, OMNI, GRASP raw + processed data
├── models/                     # Trained model weights, scalers
├── notebooks/storm_plots/      # Per-storm evaluation visualizations
├── src/
│   ├── preprocessing.py        # OMNI + GOES loading and merging
│   ├── features.py             # Feature engineering, horizon-specific feature sets
│   ├── model.py                # Multi-horizon LSTM regression
│   ├── classifier.py           # Early warning binary classifier
│   ├── ablation_model.py       # Flux-free ablation study
│   ├── storm_eval.py           # Regression model storm-level evaluation
│   ├── classifier_storm_eval.py # Classifier storm-level evaluation with persistence filtering
│   ├── storm_compare.py        # Physical signature comparison across storms
│   ├── grasp_loader.py         # GRASP/GSAT-19 data parser
│   ├── grasp_validation.py     # Cross-instrument validation
│   └── dashboard.py            # Streamlit interactive dashboard
└── README.md


## Running it

```bash
pip install -r requirements.txt
python src/preprocessing.py      # builds merged_clean.csv
python src/features.py           # builds features.csv
python src/model.py              # trains regression model
python src/classifier.py         # trains early warning classifier
python src/storm_eval.py         # storm-level regression evaluation
python src/classifier_storm_eval.py  # storm-level classifier evaluation
python src/ablation_model.py     # ablation study
python src/grasp_loader.py       # parse GRASP data
python src/grasp_validation.py   # cross-validate against GRASP
streamlit run src/dashboard.py   # launch dashboard
```

## Data sources

- GOES-15 EPEAD electron flux (>2 MeV), science-quality, 2015-2019 — NOAA NCEI
- OMNI solar wind parameters (Vsw, Bz, density, Kp, Dst), hourly, 2015-2025 — NASA OMNIWeb
- GRASP/GSAT-19 electron flux, daily files, 2017-2018 — ISRO PRADAN/ISSDC

## Team

[Add your team names here]
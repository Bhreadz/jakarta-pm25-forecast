# Jakarta PM2.5: 2024 heatmap + predicting 2025

A solo continuation of [Breathability](https://github.com/SycthePlays/Breathability), a team
hackathon app that ranks Jakarta routes by air quality. This repo asks a narrower question on
my own: **if a model learns Jakarta's air from 2024, how well does it predict the official 2025
measurements?**

- `outputs/heatmap_2024.html`: daily US AQI across Jakarta, with a day slider (green = good, red = bad)
- `outputs/heatmap_2025_compare.html`: official 2025 vs model-predicted 2025, side by side, same slider
- `outputs/model_comparison.png`, `outputs/results.csv`: error and accuracy for every model

## Results (trained on 2024, tested on all of 2025 that has data)

| Task | Model | MAE (ug/m3) | RMSE | R² | AQI category correct |
|---|---|---|---|---|---|
| **A: weather only** | Climatology (baseline) | 10.8 | 13.5 | 0.05 | 66% |
| | Ridge (linear) | 10.6 | 13.3 | 0.08 | 62% |
| | Random Forest | 10.7 | 13.4 | 0.08 | 63% |
| | LightGBM | **10.5** | 13.3 | 0.09 | 63% |
| **B: next day** | Climatology (baseline) | 10.8 | 13.6 | 0.04 | 65% |
| | Persistence "tomorrow = today" (baseline) | 7.7 | 10.2 | 0.46 | 70% |
| | Ridge (linear) | 8.0 | 10.1 | 0.48 | 70% |
| | Random Forest | **7.3** | 9.4 | 0.55 | 70% |
| | LightGBM | 8.5 | 10.8 | 0.40 | 65% |

What this says, plainly:
- **Weather alone barely beats "use last year's average for this month".** The best model gains
  0.3 ug/m3 over climatology and gets the AQI category right *less* often. Daily weather from a
  ~25 km reanalysis grid doesn't explain most of what happens at street level in Jakarta.
- **Knowing yesterday matters much more than any model.** Persistence alone cuts the error by
  ~30%. Random Forest adds a small further gain (7.7 → 7.3); LightGBM did worse than persistence.
- "Within one AQI category" is ~99% for every model, including baselines. It is not a useful
  metric here, because almost every day is Moderate or Unhealthy for Sensitive Groups.
- Per-month errors are in `outputs/monthly_mae.csv`. For the weather-only models they swing a
  lot, from ~6-8 ug/m3 in Jan and Sep 2025 up to ~16 in Oct 2025.

## The two tasks
- **Task A (main): "PM2.5 given the weather."** Features: daily temperature, humidity, rain
  (today + yesterday), pressure, wind speed and direction (as u/v), day of week, season, station.
  No past PM2.5. It uses *observed* 2025 weather, so it is not a real forecast.
- **Task B: next day.** Task A features + the measured PM2.5 of the previous 2 days.

## Method
- **Target:** daily mean PM2.5 from the two US Embassy monitors (Jakarta Central, Jakarta South),
  via the AirNow archive. Real sensors, not a model product.
- **Split strictly by time:** fit on 2024, score on 2025. Hyperparameters were chosen with
  4-fold expanding-window cross-validation inside 2024 only; 2025 was never used for tuning.
- **AQI:** US EPA PM2.5 breakpoints as revised in May 2024 (Good = 0-9.0 ug/m3), applied to
  24-hour means, which is what they are defined for. Table in `src/aqi.py`.
- **Heatmap:** each grid cell (0.01°, ~1 km) is an inverse-distance-weighted average of the
  stations reporting that day. **With only two stations, everything between them is an estimate**,
  and on days with one station the whole map takes that station's value. The grey dots are
  Breathability's district list (`Breathability_v2/lib/jakartaAreas.js`); hover for the estimate.

## Limitations (honest list)
- Only **2 stations** have a usable 2024 and 2025 record. Jakarta Central stops in April 2025, and
  both stop in early November 2025. Test set = 391 station-days.
- Open-Meteo has no boundary-layer height for Jan-Jun 2024, so that feature (important for
  pollution) is left out.
- Both stations sit in nearly the same weather-grid cell, so they get almost identical weather.
- Sources I tried that didn't work are listed in `data/SOURCES.md`.

## Run it
```bash
pip install -r requirements.txt
cd src
python fetch_airnow.py      # a few minutes, downloads 731 daily AirNow files, keeps Jakarta rows
python build_dataset.py     # weather + cleaning -> data/processed/dataset.parquet
python train_eval.py        # models + outputs/results.csv + chart
python make_heatmap.py      # needs ../Breathability_v2 next to this repo for the district dots
python aqi.py               # self-check of the AQI table
```
`forecast.py` is an earlier experiment (hourly, 1-24 h ahead, one station) and is not part of this pipeline.

## Data credits
- PM2.5: U.S. Department of State Jakarta monitors via AirNow (files.airnowtech.org).
- Weather: [Open-Meteo](https://open-meteo.com/) Historical Weather API, CC BY 4.0
  (ERA5 reanalysis, Copernicus Climate Change Service).

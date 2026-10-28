# Data sources

Downloaded 2026-10-27 (2024-2025) and 2026-10-28 (2020-2023). Raw files in `data/raw/` are kept exactly as downloaded
(only filtered to the Jakarta rows for AirNow, since the daily files cover the whole world).

| File | Source | Notes |
|---|---|---|
| `raw/airnow_jakarta_daily.dat` | AirNow archive, `https://files.airnowtech.org/airnow/YYYY/YYYYMMDD/daily_data.dat`, 2020-01-01 .. 2025-12-31 | US Department of State monitors Jakarta Central (JK1010002) and Jakarta South (JK1010001). `PM2.5-24hr` = daily mean, ug/m3. Built by `src/fetch_airnow.py`. |
| `raw/airnow_sites.dat` | `https://files.airnowtech.org/airnow/today/monitoring_site_locations.dat` | Station coordinates. |
| `raw/openmeteo_*_2020-2025.csv` | Open-Meteo Historical Weather API, `https://archive-api.open-meteo.com/v1/archive` (ERA5-based reanalysis), CC BY 4.0 | Hourly, Asia/Jakarta time, one file per station. Built by `src/build_dataset.py`. `boundary_layer_height` is empty for Jan-Jun 2024. |

## Checked and not used
- **dosairnowdata.org** (old State Dept historical CSVs): domain no longer resolves.
- **Jakarta SPKU / ISPU (Satu Data Jakarta, Kaggle mirror)**: 5 city stations, but the public files stop in Feb 2025, so no full 2025 to test against.
- **OpenAQ v3**: needs a free API key (`OPENAQ_API_KEY`). Could add more Jakarta sensors later; its Jakarta South series is the same US Embassy monitor we already have.

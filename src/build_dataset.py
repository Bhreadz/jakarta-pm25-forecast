"""Clean + join station PM2.5 with weather -> data/processed/dataset.parquet.

One row per station-day (WIB local date), 2024-01-01 .. 2025-12-31.
Run src/fetch_airnow.py first. Weather is pulled here from Open-Meteo
(cached in data/raw/ so we only download once).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# --- Stations: coordinates from AirNow's monitoring_site_locations.dat ---
sites = pd.read_csv(RAW / "airnow_sites.dat", sep="|", header=None)
sites = sites[sites[1] == "PM2.5"]
STATIONS = {r[3]: (r[8], r[9]) for r in sites.itertuples(index=False)}  # name -> (lat, lon)

# --- PM2.5: AirNow daily_data rows ---
# Columns: date | site id | site name | parameter | unit | value | averaging hours | agency
pm = pd.read_csv(RAW / "airnow_jakarta_daily.dat", sep="|", header=None,
                 names=["date", "site_id", "station", "param", "unit", "pm25", "hours", "agency"])
pm = pm[pm["param"] == "PM2.5-24hr"]
pm["date"] = pd.to_datetime(pm["date"], format="%m/%d/%y")  # AirNow daily = local calendar day
pm = pm.drop_duplicates(["station", "date"])[["station", "date", "pm25"]]

n_raw = len(pm)
bad = (pm["pm25"] < 0) | (pm["pm25"] > 500)  # sentinels (-999) and impossible values
# Flat-lined sensor: identical value 3+ days running at one station = stuck, not real.
same = pm.sort_values("date").groupby("station")["pm25"].transform(
    lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
stuck = same.reindex(pm.index) >= 3
pm = pm[~bad & ~stuck]
print(f"PM2.5 rows: {n_raw} raw, dropped {bad.sum()} out-of-range + {stuck.sum()} flat-lined "
      f"({(bad | stuck).mean():.1%})")

# Full station x day grid so missing days show up as NaN rows (not silently absent).
days = pd.date_range("2024-01-01", "2025-12-31")
grid = pd.MultiIndex.from_product([list(STATIONS), days], names=["station", "date"])
df = pm.set_index(["station", "date"]).reindex(grid).reset_index()

# --- Weather: Open-Meteo historical (ERA5-based), hourly -> daily, WIB timezone ---
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
          "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]


def weather_for(name, lat, lon):
    path = RAW / f"openmeteo_{name.replace(' ', '_')}.csv"
    if not path.exists():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
            "latitude": lat, "longitude": lon, "start_date": "2024-01-01", "end_date": "2025-12-31",
            "hourly": ",".join(HOURLY), "timezone": "Asia/Jakarta"})
        r.raise_for_status()
        pd.DataFrame(r.json()["hourly"]).to_csv(path, index=False)
    h = pd.read_csv(path, parse_dates=["time"])
    # Average wind as a vector (u, v): the mean of 350 deg and 10 deg is north, not south.
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)  # direction = where wind comes FROM
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)
    d = h.set_index("time").resample("D").agg({
        "temperature_2m": "mean", "relative_humidity_2m": "mean", "precipitation": "sum",
        "surface_pressure": "mean", "wind_speed_10m": "mean", "wind_u": "mean", "wind_v": "mean",
        "boundary_layer_height": "mean"})
    d["station"], d["lat"], d["lon"] = name, lat, lon
    return d.rename_axis("date").reset_index()


weather = pd.concat([weather_for(n, *ll) for n, ll in STATIONS.items()])
df = df.merge(weather, on=["station", "date"], how="left")

# Rain yesterday also washes out particles; it's known on the day so it's a fair feature.
df["rain_prev_day"] = df.groupby("station")["precipitation"].shift(1)

df.to_parquet(OUT / "dataset.parquet", index=False)

# --- Missing-data report ---
print("\nPM2.5 coverage (share of days with a valid value):")
print(df.assign(year=df["date"].dt.year).groupby(["station", "year"])["pm25"]
      .apply(lambda s: f"{s.notna().mean():.0%} ({s.notna().sum()} days)").unstack().to_string())
wx = df.drop(columns=["station", "date", "pm25", "lat", "lon"]).isna().mean()
print("Weather missing (share of days):", wx[wx > 0].round(3).to_dict() or "none")
# boundary_layer_height: Open-Meteo returns nothing for Jan-Jun 2024 -> not used as a feature.
print(f"Saved {len(df)} rows -> {OUT / 'dataset.parquet'}")

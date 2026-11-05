from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from aqi import CATEGORIES, category, pm25_to_aqi

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain_prev_day",
           "surface_pressure", "wind_speed_10m", "wind_u", "wind_v"]
CALENDAR = ["day_of_week", "season_sin", "season_cos", "is_south"]
FEATURES = WEATHER + CALENDAR
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
          "wind_speed_10m", "wind_direction_10m"]


def add_calendar(df):
    angle = 2 * np.pi * df["date"].dt.dayofyear / 365.25
    return df.assign(day_of_week=df["date"].dt.dayofweek, season_sin=np.sin(angle),
                     season_cos=np.cos(angle), is_south=(df["station"] == "Jakarta South").astype(int))


def forecast_weather(name, lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY),
        "timezone": "Asia/Jakarta", "past_days": 1, "forecast_days": 8})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)
    d = h.set_index("time").resample("D").agg({
        "temperature_2m": "mean", "relative_humidity_2m": "mean", "precipitation": "sum",
        "surface_pressure": "mean", "wind_speed_10m": "mean", "wind_u": "mean", "wind_v": "mean"})
    d["rain_prev_day"] = d["precipitation"].shift(1)
    d["station"] = name
    return d.rename_axis("date").reset_index()


def cams_forecast(lat, lon):
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=60, params={
        "latitude": lat, "longitude": lon, "hourly": "pm2_5",
        "timezone": "Asia/Jakarta", "forecast_days": 7})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")["pm2_5"].resample("D").mean()


history = add_calendar(pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet"))
history = history.dropna(subset=["pm25"] + FEATURES).sort_values(["date", "station"])
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                          min_child_samples=30, random_state=0, verbose=-1)
model.fit(history[FEATURES], history["pm25"])
print(f"Trained on {len(history)} measured station-days, "
      f"{history['date'].min():%Y-%m-%d} to {history['date'].max():%Y-%m-%d}")

stations = history.groupby("station")[["lat", "lon"]].first()
today = pd.Timestamp.now(tz="Asia/Jakarta").normalize().tz_localize(None)

rows = []
for name, (lat, lon) in stations.iterrows():
    w = add_calendar(forecast_weather(name, lat, lon))
    w = w[w["date"] > today].dropna(subset=FEATURES)
    w["predicted_pm25"] = model.predict(w[FEATURES])
    w["cams_pm25"] = w["date"].map(cams_forecast(lat, lon))
    rows.append(w)

out = pd.concat(rows)[["date", "station", "predicted_pm25", "cams_pm25"]]
out["predicted_aqi"] = pm25_to_aqi(out["predicted_pm25"])
out["predicted_category"] = [CATEGORIES[c] for c in category(out["predicted_pm25"])]
out = out.sort_values(["date", "station"]).round({"predicted_pm25": 1, "cams_pm25": 1})
out.to_csv(ROOT / "outputs" / "forecast.csv", index=False)

tomorrow = out[out["date"] == out["date"].min()]
print(f"\nTomorrow, {tomorrow['date'].iloc[0]:%A %d %B %Y}:")
for r in tomorrow.itertuples():
    print(f"  {r.station}: {r.predicted_pm25} ug/m3, AQI {r.predicted_aqi:.0f} ({r.predicted_category})"
          f"  | CAMS says {r.cams_pm25} ug/m3")
print("\nNext 7 days:")
print(out.assign(date=out["date"].dt.strftime("%a %d %b")).to_string(index=False))
print(f"\nSaved {ROOT / 'outputs' / 'forecast.csv'}")

import os
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

LAT, LON = -6.2088, 106.8456
START = "2023-01-01"
END = (date.today() - timedelta(days=2)).isoformat()
HORIZONS = [1, 3, 6, 12, 24]
STATION_ID = 5077558
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

BREAKS = [9.0, 35.4, 55.4, 125.4, 225.4]
CATS = ["Good", "Moderate", "Unhealthy (sensitive)", "Unhealthy", "Very unhealthy", "Hazardous"]
SENSITIVE = 35.4


def cached(name, fetch):
    path = DATA / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    df.to_csv(path)
    return df


def fetch_openaq(key):
    s = requests.Session()
    s.headers["X-API-Key"] = key
    sensor = STATION_ID
    if sensor is None:
        r = s.get("https://api.openaq.org/v3/locations",
                  params={"coordinates": f"{LAT},{LON}", "radius": 25000, "limit": 100})
        r.raise_for_status()
        cands = []
        for loc in r.json()["results"]:
            first, last = loc.get("datetimeFirst"), loc.get("datetimeLast")
            for sen in loc["sensors"]:
                if sen["parameter"]["name"] == "pm25" and first and last:
                    span = pd.Timestamp(last["utc"]) - pd.Timestamp(first["utc"])
                    cands.append((span, sen["id"], loc["name"], first["utc"][:10], last["utc"][:10]))
        cands.sort(reverse=True)
        print("PM2.5 sensors near Jakarta (longest history first):")
        for span, sid, name, f, l in cands[:10]:
            print(f"  sensor {sid:>8}  {name:<35} {f} -> {l}")
        sensor = cands[0][1]

    rows = []
    edges = pd.date_range(START, END, freq="30D").append(pd.DatetimeIndex([END]))
    for a, b in zip(edges[:-1], edges[1:]):
        for attempt in range(4):
            r = s.get(f"https://api.openaq.org/v3/sensors/{sensor}/hours",
                      params={"datetime_from": a.isoformat(), "datetime_to": b.isoformat(), "limit": 1000},
                      timeout=60)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        rows += [(m["period"]["datetimeFrom"]["utc"], m["value"]) for m in r.json()["results"]]
        print(f"  {a:%Y-%m-%d}: {len(rows)} hours so far", end="\r")
        time.sleep(1.1)
    df = pd.DataFrame(rows, columns=["time", "pm25"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.floor("h")
    return df.groupby("time").mean()


def fetch_open_meteo(url, variables):
    r = requests.get(url, params={"latitude": LAT, "longitude": LON, "start_date": START,
                                  "end_date": END, "hourly": ",".join(variables), "timezone": "GMT"},
                     timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    return df.set_index(pd.to_datetime(df.pop("time")))


WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]
weather = cached("weather", lambda: fetch_open_meteo(
    "https://archive-api.open-meteo.com/v1/archive", WEATHER_VARS))
cams = cached("cams", lambda: fetch_open_meteo(
    "https://air-quality-api.open-meteo.com/v1/air-quality", ["pm2_5"]))["pm2_5"]

key = os.environ.get("OPENAQ_API_KEY")
if key:
    pm = cached("openaq", lambda: fetch_openaq(key))["pm25"]
else:
    print("!! OPENAQ_API_KEY not set -- target is CAMS model PM2.5, not a real station. "
          "Fine for testing the pipeline, not for the write-up.")
    pm = cams.copy()

idx = pd.date_range(max(pm.index.min(), weather.index.min()), min(pm.index.max(), weather.index.max()), freq="h")
pm = pm.reindex(idx)
pm[(pm < 0) | (pm > 500)] = np.nan
print(f"{len(idx)} hours, {pm.isna().mean():.1%} missing PM2.5")
pm_filled = pm.ffill(limit=3)
weather = weather.reindex(idx).ffill(limit=3)

X = pd.DataFrame(index=idx)
for lag in [0, 1, 2, 3, 6, 12, 24]:
    X[f"pm_lag{lag}"] = pm_filled.shift(lag)
for w in [3, 6, 24]:
    X[f"pm_mean{w}"] = pm_filled.rolling(w, min_periods=1).mean()
X["pm_diff1"] = X["pm_lag0"] - X["pm_lag1"]
for v in WEATHER_VARS:
    if v != "wind_direction_10m":
        X[v] = weather[v]
rad = np.deg2rad(weather["wind_direction_10m"])
X["wind_u"], X["wind_v"] = np.sin(rad), np.cos(rad)
X["rain_24h"] = weather["precipitation"].rolling(24, min_periods=1).sum()


def calendar(ts):
    wib = ts + pd.Timedelta(hours=7)
    return pd.DataFrame({"hour": wib.hour, "weekday": wib.weekday, "month": wib.month,
                         "dry_season": wib.month.isin([5, 6, 7, 8, 9, 10]).astype(int)}, index=ts)


def category(v):
    return np.digitize(v, BREAKS, right=True)


results = []
for h in HORIZONS:
    target_time = idx + pd.Timedelta(hours=h)
    df = X.join(calendar(target_time).set_axis(idx).add_prefix("tgt_"))
    df["y"] = pm.shift(-h)
    df["persistence"] = X["pm_lag0"]
    df["cams"] = cams.reindex(target_time).to_numpy()
    df = df.dropna(subset=["y", "pm_lag0"])

    cut = df.index[int(len(df) * 0.8)]
    train, test = df[df.index < cut], df[df.index >= cut]
    feats = [c for c in df.columns if c not in ("y", "persistence", "cams")]

    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    val = train.index >= train.index[int(len(train) * 0.85)]
    model.fit(train.loc[~val, feats], train.loc[~val, "y"],
              eval_X=train.loc[val, feats], eval_y=train.loc[val, "y"],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    test = test.assign(lgbm=model.predict(test[feats]))

    y = test["y"].to_numpy()
    bad = y > SENSITIVE
    for name in ["persistence", "cams", "lgbm"]:
        p = test[name].to_numpy()
        ok = ~np.isnan(p)
        if not key and name == "cams":
            continue
        results.append({
            "horizon_h": h, "model": name,
            "MAE": np.abs(p[ok] - y[ok]).mean(),
            "cat_acc": (category(p[ok]) == category(y[ok])).mean(),
            "bad_recall": (p[ok & bad] > SENSITIVE).mean() if (ok & bad).any() else np.nan,
            "bad_precision": (y[ok & (p > SENSITIVE)] > SENSITIVE).mean() if (ok & (p > SENSITIVE)).any() else np.nan,
        })

res = pd.DataFrame(results)
print(f"\nTest period: {cut:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}   ({(pm.loc[cut:] > SENSITIVE).mean():.0%} of hours above {SENSITIVE})")
print(res.pivot(index="horizon_h", columns="model", values=["MAE", "bad_recall"]).round(2).to_string())
res.to_csv(DATA / "results.csv", index=False)

assert list(category(np.array([0, 9.0, 9.1, 35.4, 35.5, 55.5, 300]))) == [0, 0, 1, 1, 2, 3, 5]

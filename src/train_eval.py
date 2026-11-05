from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aqi import category

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 0

df = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet").sort_values(["date", "station"])

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
doy = 2 * np.pi * df["date"].dt.dayofyear / 365.25
df["season_sin"], df["season_cos"] = np.sin(doy), np.cos(doy)
df["is_south"] = (df["station"] == "Jakarta South").astype(int)
g = df.groupby("station")["pm25"]
df["pm_yesterday"] = g.shift(1)
df["pm_2days_ago"] = g.shift(2)

WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain_prev_day",
           "surface_pressure", "wind_speed_10m", "wind_u", "wind_v"]
CALENDAR = ["day_of_week", "season_sin", "season_cos", "is_south"]
TASKS = {"A_weather_only": WEATHER + CALENDAR,
         "B_next_day": WEATHER + CALENDAR + ["pm_yesterday", "pm_2days_ago"]}

MODELS = {
    "Ridge (linear)": (make_pipeline(StandardScaler(), Ridge()),
                       {"ridge__alpha": [0.1, 1, 10, 100]}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"max_depth": [4, 8, None], "min_samples_leaf": [1, 5, 20]}),
    "LightGBM": (lgb.LGBMRegressor(n_estimators=300, random_state=SEED, verbose=-1),
                 {"learning_rate": [0.03, 0.1], "num_leaves": [7, 15, 31], "min_child_samples": [10, 30]}),
}


def score(y, p):
    cy, cp = category(y), category(p)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "AQI_category_acc": (cy == cp).mean(),
            "within_1_category": (abs(cy - cp) <= 1).mean()}


results, monthly, preds = [], [], []
for task, feats in TASKS.items():
    d = df.dropna(subset=["pm25"] + feats)
    train, test = d[d["date"].dt.year <= 2024], d[d["date"].dt.year == 2025]
    print(f"\n=== {task}: {len(train)} train days (2020-2024), {len(test)} test days (2025) ===")

    out = test[["station", "date", "month", "lat", "lon", "pm25"]].copy()
    clim = train.groupby(["station", "month"])["pm25"].mean().rename("Climatology (2020-24 monthly mean)")
    out = out.join(clim, on=["station", "month"])
    if "pm_yesterday" in feats:
        out["Persistence (yesterday)"] = test["pm_yesterday"]

    for name, (model, grid) in MODELS.items():
        search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=4),
                              scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(train[feats], train["pm25"])
        out[name] = search.predict(test[feats])
        print(f"  {name:<15} best params {search.best_params_}  (CV MAE {-search.best_score_:.1f})")

    model_cols = [c for c in out.columns if c not in ("station", "date", "month", "lat", "lon", "pm25")]
    for name in model_cols:
        results.append({"task": task, "model": name, **score(out["pm25"], out[name])})
        m = out.groupby(out["date"].dt.month).apply(lambda x: mean_absolute_error(x["pm25"], x[name]))
        monthly.append(m.rename(f"{task} | {name}"))
    preds.append(out.assign(task=task))

res = pd.DataFrame(results)
res.to_csv(OUT / "results.csv", index=False)
pd.concat(monthly, axis=1).rename_axis("month_2025").round(1).to_csv(OUT / "monthly_mae.csv")
pd.concat(preds).to_parquet(OUT / "predictions_2025.parquet", index=False)
pd.set_option("display.width", 160)
print("\n", res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
for ax, (task, r) in zip(axes, res.groupby("task", sort=False)):
    r = r.sort_values("MAE", ascending=False)
    colors = ["#9e9e9e" if ("Climatology" in m or "Persistence" in m) else "#1f77b4" for m in r["model"]]
    ax.barh(r["model"], r["MAE"], color=colors)
    for y, (mae, acc) in enumerate(zip(r["MAE"], r["AQI_category_acc"])):
        ax.text(mae, y, f" {mae:.1f}  ({acc:.0%} AQI cat.)", va="center", fontsize=9)
    ax.set_title({"A_weather_only": "Task A: weather only (no past PM2.5)",
                  "B_next_day": "Task B: next day (knows yesterday)"}[task])
    ax.set_xlim(0, res["MAE"].max() * 1.6)
    ax.set_xlabel("Mean absolute error on 2025, ug/m3 (lower = better)")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Trained on 2020-2024, tested on 2025 - Jakarta US Embassy PM2.5 (grey = baseline)")
fig.savefig(OUT / "model_comparison.png", dpi=150)
print(f"\nSaved results to {OUT}")

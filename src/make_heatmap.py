"""Daily US-AQI heatmaps of Jakarta with a time slider (single self-contained HTML files).

outputs/heatmap_2024.html          measured 2024, one frame per day
outputs/heatmap_2025_compare.html  2025: official (measured) vs model-predicted, side by side

Values between stations are ESTIMATES: each grid cell is an inverse-distance-
weighted (IDW) average of the stations, so closer stations count more.
District dots are the kecamatan list from the Breathability app
(Breathability_v2/lib/jakartaAreas.js), so the map lines up with the app's areas.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aqi import BREAKPOINTS, COLORS, pm25_to_aqi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LAT = np.arange(-6.40, -6.05 + 1e-9, 0.01)   # bounding box: Jakarta + near suburbs
LON = np.arange(106.65, 107.00 + 1e-9, 0.01)
GLAT, GLON = np.meshgrid(LAT, LON, indexing="ij")

# Fixed colour scale for every frame (otherwise colours would mean different things
# on different days). Stepped bands, AQI 0-300; anything above 300 shows as the top colour.
ZMAX = 300
edges = [0] + [b[3] for b in BREAKPOINTS[:5]]          # 0, 50, 100, 150, 200, 300
SCALE = []
for (lo, hi), c in zip(zip(edges[:-1], edges[1:]), COLORS):
    SCALE += [[lo / ZMAX, c], [hi / ZMAX, c]]
COLORBAR = dict(title="US AQI", tickvals=edges, len=0.8)

# Breathability districts inside the box (parsed straight from the app's JS file).
js = (ROOT.parent / "Breathability_v2" / "lib" / "jakartaAreas.js").read_text(encoding="utf-8")
areas = pd.DataFrame(re.findall(r"name: '([^']+)'.*?lat: (-?[\d.]+), lon: (-?[\d.]+)", js),
                     columns=["name", "lat", "lon"]).astype({"lat": float, "lon": float})
areas = areas[areas["lat"].between(LAT[0], LAT[-1]) & areas["lon"].between(LON[0], LON[-1])]


def idw(lats, lons, values, qlat, qlon, power=2):
    """Inverse-distance weighting. Degrees are fine as distance at this scale (~40 km)."""
    d = np.hypot(qlat[..., None] - lats, qlon[..., None] - lons)
    w = 1 / np.maximum(d, 1e-6) ** power
    return (w * values).sum(-1) / w.sum(-1)


def day_traces(day_rows, col):
    """Heatmap + district dots + station markers for one day and one value column."""
    r = day_rows.dropna(subset=[col])
    if r.empty:  # no station reported that day
        z, area_aqi = np.full(GLAT.shape, np.nan), np.full(len(areas), np.nan)
    else:
        args = r["lat"].to_numpy(), r["lon"].to_numpy(), r[col].to_numpy()
        z = pm25_to_aqi(idw(*args, GLAT, GLON))
        area_aqi = pm25_to_aqi(idw(*args, areas["lat"].to_numpy(), areas["lon"].to_numpy()))
    st_aqi = pm25_to_aqi(r[col])
    return [
        go.Heatmap(z=z, x=LON, y=LAT, zmin=0, zmax=ZMAX, colorscale=SCALE, colorbar=COLORBAR,
                   hovertemplate="lat %{y:.2f}, lon %{x:.2f}<br>estimated AQI %{z}<extra></extra>"),
        go.Scatter(x=areas["lon"], y=areas["lat"], mode="markers",
                   marker=dict(size=5, color="rgba(0,0,0,0.35)"),
                   text=[f"{n}: est. AQI {a:.0f}" for n, a in zip(areas["name"], area_aqi)],
                   hoverinfo="text", showlegend=False),
        go.Scatter(x=r["lon"], y=r["lat"], mode="markers+text",
                   marker=dict(size=14, symbol="star", color="white", line=dict(width=1.5, color="black")),
                   text=[f"{s}<br>{a:.0f}" for s, a in zip(r["station"], st_aqi)],
                   textposition="top center", hoverinfo="text", showlegend=False),
    ]


def build(df, panels, title, path):
    """panels = [(column, subplot title), ...]; one slider frame per date."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels],
                        horizontal_spacing=0.06)
    dates = sorted(df["date"].unique())
    by_day = dict(tuple(df.groupby("date")))

    def traces(day):
        out = []
        for i, (col, _) in enumerate(panels):
            t = day_traces(by_day[day], col)
            if i > 0:
                t[0].showscale = False
            out += t
        return out

    for i, t in enumerate(traces(dates[0])):
        fig.add_trace(t, row=1, col=i // 3 + 1)
    fig.frames = [go.Frame(data=traces(d), name=f"{pd.Timestamp(d):%Y-%m-%d}") for d in dates]
    labels = [f.name for f in fig.frames]
    fig.update_layout(
        title=title, height=700, margin=dict(t=120, b=110), plot_bgcolor="#f4f4f4",
        sliders=[dict(active=0, currentvalue=dict(prefix="Day: "), pad=dict(t=40), steps=[
            dict(label=l, method="animate",
                 args=[[l], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for l in labels])],
        updatemenus=[dict(type="buttons", direction="left", x=1, y=1.13, xanchor="right", buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        annotations=list(fig.layout.annotations) + [dict(
            text="Stars = real monitors. Everything else is interpolated between them (IDW), "
                 "so it is an estimate. Dots = Breathability districts.",
            xref="paper", yref="paper", x=0, y=1.09, xanchor="left", showarrow=False, font=dict(size=11))])
    for i in range(len(panels)):
        fig.update_xaxes(title="longitude", range=[LON[0], LON[-1]], constrain="domain", row=1, col=i + 1)
        fig.update_yaxes(title="latitude" if i == 0 else None, scaleanchor=f"x{i + 1 if i else ''}",
                         row=1, col=i + 1)
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    print("wrote", path)


if __name__ == "__main__":
    assert np.isclose(idw(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([10.0, 20.0]),
                          np.array(0.5), np.array(0.0)), 15)  # midpoint = plain average
    data = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    build(data[data["date"].dt.year == 2024], [("pm25", "Measured")],
          "Jakarta air quality 2024 (daily US AQI from PM2.5)", OUT / "heatmap_2024.html")

    res = pd.read_csv(OUT / "results.csv")
    best = res[res["task"] == "A_weather_only"].sort_values("MAE").iloc[0]["model"]
    pred = pd.read_parquet(OUT / "predictions_2025.parquet")
    pred = pred[pred["task"] == "A_weather_only"]
    # Keep every 2025 day on the slider, including days a station had no official value.
    base = data[data["date"].dt.year == 2025][["station", "date", "lat", "lon", "pm25"]]
    comp = base.merge(pred[["station", "date", best]], on=["station", "date"], how="left")
    build(comp, [("pm25", "Official 2025 (measured)"), (best, f"Predicted: {best}, trained on 2024")],
          f"2025: official vs predicted (Task A, weather only) - best model by MAE: {best}",
          OUT / "heatmap_2025_compare.html")

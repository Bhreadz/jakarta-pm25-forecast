import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots

from aqi import COLORS, pm25_to_aqi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STEP = 0.005
LAT = np.arange(-6.40, -6.05 + 1e-9, STEP)
LON = np.arange(106.65, 107.00 + 1e-9, STEP)
GLAT, GLON = np.meshgrid(LAT, LON, indexing="ij")
CENTER = dict(lat=(LAT[0] + LAT[-1]) / 2, lon=(LON[0] + LON[-1]) / 2)
CORNERS = [[LON[0], LAT[-1]], [LON[-1], LAT[-1]], [LON[-1], LAT[0]], [LON[0], LAT[0]]]

SCALE = [[i / 3, c] for i, c in enumerate(COLORS[:4])]
CMAP = LinearSegmentedColormap.from_list("aqi", [(f, c) for f, c in SCALE])

js = (ROOT.parent / "Breathability_v2" / "lib" / "jakartaAreas.js").read_text(encoding="utf-8")
areas = pd.DataFrame(re.findall(r"name: '([^']+)'.*?lat: (-?[\d.]+), lon: (-?[\d.]+)", js),
                     columns=["name", "lat", "lon"]).astype({"lat": float, "lon": float})
areas = areas[areas["lat"].between(LAT[0], LAT[-1]) & areas["lon"].between(LON[0], LON[-1])]


def idw(lats, lons, values, qlat, qlon, power=2):
    d = np.hypot(qlat[..., None] - lats, qlon[..., None] - lons)
    w = 1 / np.maximum(d, 1e-6) ** power
    return (w * values).sum(-1) / w.sum(-1)


def to_png(aqi_grid, lo, hi):
    rgba = CMAP((np.flipud(aqi_grid) - lo) / (hi - lo))
    rgba[..., 3] *= EDGE_FADE
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_ramp = lambda n: np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.15 * n), 0, 1)
EDGE_FADE = np.outer(_ramp(len(LAT)), _ramp(len(LON)))


def image_layer(png):
    return dict(sourcetype="image", source=png, coordinates=CORNERS, opacity=0.6, below="traces")


def day_frame(day_rows, col, map_id, rng, first=True):
    r = day_rows.dropna(subset=[col])
    if r.empty:
        png, area_aqi, st_aqi = BLANK, np.full(len(areas), np.nan), []
    else:
        args = r["lat"].to_numpy(), r["lon"].to_numpy(), r[col].to_numpy()
        png = to_png(pm25_to_aqi(idw(*args, GLAT, GLON)), *rng)
        area_aqi = pm25_to_aqi(idw(*args, areas["lat"].to_numpy(), areas["lon"].to_numpy()))
        st_aqi = pm25_to_aqi(r[col])
    area_text = [f"{n}: est. AQI {a:.0f}" for n, a in zip(areas["name"], area_aqi)]
    st_text = [f"{s}: {a:.0f}" for s, a in zip(r["station"], st_aqi)]
    if not first:
        return [go.Scattermap(text=area_text),
                go.Scattermap(lat=r["lat"], lon=r["lon"], text=st_text)], image_layer(png)
    return [
        go.Scattermap(lat=areas["lat"], lon=areas["lon"], mode="markers", subplot=map_id,
                      marker=dict(size=6, color="rgba(40,40,40,0.6)"), text=area_text,
                      hoverinfo="text", showlegend=False),
        go.Scattermap(lat=r["lat"], lon=r["lon"], mode="markers+text", subplot=map_id,
                      marker=dict(size=14, color="#1a1a1a"), text=st_text, textposition="top right",
                      textfont=dict(size=13, color="#1a1a1a"), hoverinfo="text", showlegend=False),
    ], image_layer(png)


BLANK = to_png(np.full(GLAT.shape, np.nan), 0, 1)


def build(df, panels, title, path=None):
    rows = 2 if len(panels) == 4 else 1
    cols = len(panels) // rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p[1] for p in panels],
                        specs=[[{"type": "map"}] * cols] * rows,
                        horizontal_spacing=0.02, vertical_spacing=0.05)
    dates = sorted(df["date"].unique())
    by_day = dict(tuple(df.groupby("date")))
    map_ids = ["map" if i == 0 else f"map{i + 1}" for i in range(len(panels))]
    zoom = 10.4 if len(panels) == 1 else 9.7
    aqi = pm25_to_aqi(df[[c for c, _ in panels]].to_numpy())
    rng = (np.nanmin(aqi), np.nanmax(aqi))
    print(f"  colour range: AQI {rng[0]:.0f} to {rng[1]:.0f}")

    maps = {}
    for (col, _), m in zip(panels, map_ids):
        traces, layer = day_frame(by_day[dates[0]], col, m, rng)
        fig.add_traces(traces)
        maps[m] = dict(style="carto-positron", center=CENTER, zoom=zoom, layers=[layer])

    frames = []
    for d in dates:
        data, layout = [], {}
        for (col, _), m in zip(panels, map_ids):
            traces, layer = day_frame(by_day[d], col, m, rng, first=False)
            data += traces
            layout[m] = dict(layers=[layer])
        frames.append(go.Frame(name=f"{pd.Timestamp(d):%Y-%m-%d}", data=data,
                               traces=list(range(len(data))), layout=layout))
    fig.frames = frames
    labels = [f.name for f in frames]
    fig.update_layout(
        maps, height=720 if rows == 1 else 1000,
        title=dict(text=title, subtitle=dict(
            text="Black dots = real monitors. Colours between them are estimates (IDW). "
                 "Grey dots = Breathability districts, hover for values.")),
        margin=dict(t=110, b=110, l=10, r=10),
        sliders=[dict(active=0, currentvalue=dict(prefix="Day: "), pad=dict(t=20), steps=[
            dict(label=l, method="animate",
                 args=[[l], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for l in labels])],
        updatemenus=[dict(type="buttons", direction="left", x=1, y=1.1, xanchor="right", yanchor="bottom", buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0))])])])
    if path:
        fig.write_html(path, include_plotlyjs="cdn", auto_play=False, config={"displayModeBar": False})
        print("wrote", path)
    return fig


if __name__ == "__main__":
    assert np.isclose(idw(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([10.0, 20.0]),
                          np.array(0.5), np.array(0.0)), 15)
    data = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    build(data[data["date"].dt.year == 2024], [("pm25", "Measured")],
          "Jakarta air quality 2024 (daily US AQI from PM2.5)", OUT / "heatmap_2024.html")

    res = pd.read_csv(OUT / "results.csv")
    mae = res[res["task"] == "A_weather_only"].set_index("model")["MAE"]
    models = ["Ridge (linear)", "Random Forest", "LightGBM"]
    pred = pd.read_parquet(OUT / "predictions_2025.parquet")
    pred = pred[pred["task"] == "A_weather_only"]
    base = data[data["date"].dt.year == 2025][["station", "date", "lat", "lon", "pm25"]]
    comp = base.merge(pred[["station", "date"] + models], on=["station", "date"], how="left")
    build(comp, [("pm25", "Official 2025 (measured)")]
          + [(m, f"{m}: off by {mae[m]:.1f} ug/m3 on average") for m in models],
          "2025: official vs all 3 models (weather only, trained on 2020-2024)",
          OUT / "heatmap_2025_compare.html")

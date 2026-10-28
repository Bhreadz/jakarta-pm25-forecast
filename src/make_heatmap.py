"""Daily US-AQI heatmaps of Jakarta with a time slider (single self-contained HTML files).

outputs/heatmap_2024.html          measured 2024, one frame per day
outputs/heatmap_2025_compare.html  2025: official (measured) vs model-predicted, side by side

Values between stations are ESTIMATES: each grid cell is an inverse-distance-
weighted (IDW) average of the stations, so closer stations count more.
District dots are the kecamatan list from the Breathability app
(Breathability_v2/lib/jakartaAreas.js), so the map lines up with the app's areas.
"""
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

from aqi import BREAKPOINTS, COLORS, pm25_to_aqi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STEP = 0.005                                   # ~550 m grid; the map blends between cells
LAT = np.arange(-6.40, -6.05 + 1e-9, STEP)    # bounding box: Jakarta + near suburbs
LON = np.arange(106.65, 107.00 + 1e-9, STEP)
GLAT, GLON = np.meshgrid(LAT, LON, indexing="ij")
CENTER = dict(lat=(LAT[0] + LAT[-1]) / 2, lon=(LON[0] + LON[-1]) / 2)
# Image corners (top-left, top-right, bottom-right, bottom-left) as [lon, lat].
CORNERS = [[LON[0], LAT[-1]], [LON[-1], LAT[-1]], [LON[-1], LAT[0]], [LON[0], LAT[0]]]

# Fixed colour scale for every frame (otherwise colours would mean different things on
# different days). Smooth gradient: each AQI band's colour sits at the middle of its band
# (Good = green at AQI 25, Moderate = yellow at 75, ...) and blends into the next.
ZMAX = 300
edges = [0] + [b[3] for b in BREAKPOINTS[:5]]          # 0, 50, 100, 150, 200, 300
mids = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
SCALE = [[0, COLORS[0]]] + [[m / ZMAX, c] for m, c in zip(mids, COLORS)] + [[1, COLORS[4]]]
CMAP = LinearSegmentedColormap.from_list("aqi", [(f, c) for f, c in SCALE])
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


def to_png(aqi_grid):
    """Grid of AQI values -> small PNG (as a data URI) the map draws over the streets.
    Row 0 of an image is the top, i.e. the northernmost latitude, so flip it."""
    rgba = CMAP(np.flipud(aqi_grid) / ZMAX)
    rgba[..., 3] *= EDGE_FADE          # soft edges instead of a hard rectangle
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Opacity 1 in the middle, fading to 0 over the outer ~15% of the box on every side.
_ramp = lambda n: np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.15 * n), 0, 1)
EDGE_FADE = np.outer(_ramp(len(LAT)), _ramp(len(LON)))


def image_layer(png):
    # "below" puts the colours under the street names so the labels stay readable.
    return dict(sourcetype="image", source=png, coordinates=CORNERS, opacity=0.6, below="traces")


def day_frame(day_rows, col, map_id, first=True):
    """Colour image + district dots + station markers for one day and one value column.
    Frames (first=False) only carry what changes from day to day."""
    r = day_rows.dropna(subset=[col])
    if r.empty:  # no station reported that day: clear image, no estimates
        png, area_aqi, st_aqi = BLANK, np.full(len(areas), np.nan), []
    else:
        args = r["lat"].to_numpy(), r["lon"].to_numpy(), r[col].to_numpy()
        png = to_png(pm25_to_aqi(idw(*args, GLAT, GLON)))
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


BLANK = to_png(np.full(GLAT.shape, np.nan))  # NaN -> fully transparent


def build(df, panels, title, path):
    """panels = [(column, subplot title), ...]; one slider frame per date."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels],
                        specs=[[{"type": "map"}] * len(panels)], horizontal_spacing=0.02)
    dates = sorted(df["date"].unique())
    by_day = dict(tuple(df.groupby("date")))
    map_ids = ["map" if i == 0 else f"map{i + 1}" for i in range(len(panels))]
    zoom = 10.4 if len(panels) == 1 else 9.7

    maps = {}
    for (col, _), m in zip(panels, map_ids):
        traces, layer = day_frame(by_day[dates[0]], col, m)
        fig.add_traces(traces)
        maps[m] = dict(style="carto-positron", center=CENTER, zoom=zoom, layers=[layer])
    # The image has no colour bar of its own, so an invisible point carries the legend.
    fig.add_trace(go.Scattermap(lat=[CENTER["lat"]], lon=[CENTER["lon"]], mode="markers",
                                marker=dict(size=0, opacity=0, color=[0], colorscale=SCALE, cmin=0,
                                            cmax=ZMAX, showscale=True, colorbar=COLORBAR),
                                hoverinfo="skip", showlegend=False))

    frames = []
    for d in dates:
        data, layout = [], {}
        for (col, _), m in zip(panels, map_ids):
            traces, layer = day_frame(by_day[d], col, m, first=False)
            data += traces
            layout[m] = dict(layers=[layer])
        frames.append(go.Frame(name=f"{pd.Timestamp(d):%Y-%m-%d}", data=data,
                               traces=list(range(len(data))), layout=layout))
    fig.frames = frames
    labels = [f.name for f in frames]
    fig.update_layout(
        maps, height=720,
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

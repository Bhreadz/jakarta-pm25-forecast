"""PM2.5 (ug/m3, 24-hour mean) -> US AQI.

Breakpoints: US EPA PM2.5 NAAQS revision, effective 6 May 2024
(Good now ends at 9.0 ug/m3 instead of 12.0). Source: EPA AQI Technical
Assistance Document, May 2024 (airnow.gov/publications).
These are defined for 24-hour averages, which is exactly what we use
(daily means), so no hourly approximation is needed.

    conc_lo  conc_hi  aqi_lo aqi_hi  category
"""
import numpy as np

BREAKPOINTS = [
    (0.0,     9.0,    0,  50, "Good"),
    (9.1,    35.4,   51, 100, "Moderate"),
    (35.5,   55.4,  101, 150, "Unhealthy for Sensitive Groups"),
    (55.5,  125.4,  151, 200, "Unhealthy"),
    (125.5, 225.4,  201, 300, "Very Unhealthy"),
    (225.5, 325.4,  301, 500, "Hazardous"),
]
CATEGORIES = [b[4] for b in BREAKPOINTS]
# Standard AQI band colours (green -> maroon), used by the heatmap.
COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]


def pm25_to_aqi(pm):
    """Linear interpolation inside the band the (truncated) concentration falls in."""
    pm = np.floor(np.asarray(pm, dtype=float) * 10) / 10  # EPA truncates to 0.1
    aqi = np.full(pm.shape, np.nan)
    for lo, hi, alo, ahi, _ in BREAKPOINTS:
        m = (pm >= lo) & (pm <= hi)
        aqi[m] = (ahi - alo) / (hi - lo) * (pm[m] - lo) + alo
    aqi[pm > 325.4] = 500  # off the scale
    return np.round(aqi)


def category(pm):
    """Category index 0-5 (0 = Good). Works on concentrations, not AQI."""
    pm = np.floor(np.asarray(pm, dtype=float) * 10) / 10
    return np.digitize(pm, [b[1] for b in BREAKPOINTS[:-1]], right=True)


if __name__ == "__main__":
    # Spot checks against the EPA table edges.
    assert list(pm25_to_aqi([0, 9.0, 9.1, 35.4, 35.5, 55.4, 125.4, 400])) == [0, 50, 51, 100, 101, 150, 200, 500]
    assert list(category([0, 9.0, 9.05, 35.4, 35.5, 55.5, 300])) == [0, 0, 0, 1, 2, 3, 5]
    print("aqi.py ok")

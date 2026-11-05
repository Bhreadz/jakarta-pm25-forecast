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
COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]


def pm25_to_aqi(pm):
    pm = np.floor(np.asarray(pm, dtype=float) * 10) / 10
    aqi = np.full(pm.shape, np.nan)
    for lo, hi, alo, ahi, _ in BREAKPOINTS:
        m = (pm >= lo) & (pm <= hi)
        aqi[m] = (ahi - alo) / (hi - lo) * (pm[m] - lo) + alo
    aqi[pm > 325.4] = 500
    return np.round(aqi)


def category(pm):
    pm = np.floor(np.asarray(pm, dtype=float) * 10) / 10
    return np.digitize(pm, [b[1] for b in BREAKPOINTS[:-1]], right=True)


if __name__ == "__main__":
    assert list(pm25_to_aqi([0, 9.0, 9.1, 35.4, 35.5, 55.4, 125.4, 400])) == [0, 50, 51, 100, 101, 150, 200, 500]
    assert list(category([0, 9.0, 9.05, 35.4, 35.5, 55.5, 300])) == [0, 0, 0, 1, 2, 3, 5]
    print("aqi.py ok")

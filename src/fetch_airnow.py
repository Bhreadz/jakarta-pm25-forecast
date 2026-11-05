import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "airnow_jakarta_daily.dat"
URL = "https://files.airnowtech.org/airnow/{d:%Y}/{d:%Y%m%d}/daily_data.dat"


def jakarta_rows(day):
    for attempt in range(3):
        try:
            r = requests.get(URL.format(d=day), timeout=60)
            if r.status_code == 200:
                return [l for l in r.text.splitlines() if "Jakarta" in l]
            print(day.date(), "HTTP", r.status_code)
            return []
        except requests.RequestException:
            time.sleep(5)
    print("gave up on", day.date())
    return []


sites = requests.get("https://files.airnowtech.org/airnow/today/monitoring_site_locations.dat", timeout=60)
(RAW / "airnow_sites.dat").write_text("\n".join(l for l in sites.text.splitlines() if "Jakarta" in l))

done = set()
if OUT.exists():
    done = {line.split("|")[0] for line in OUT.read_text().splitlines()}
todo = [d for d in pd.date_range("2020-01-01", "2025-12-31") if d.strftime("%m/%d/%y") not in done]

with OUT.open("a") as f, ThreadPoolExecutor(8) as pool:
    for day, rows in zip(todo, pool.map(jakarta_rows, todo)):
        f.write("".join(l + "\n" for l in rows))
        f.flush()
        print(day.date(), len(rows), "rows", flush=True)

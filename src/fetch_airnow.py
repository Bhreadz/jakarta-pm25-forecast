"""Download US Embassy Jakarta PM2.5 daily means (2024-2025) from the AirNow archive.

AirNow publishes one `daily_data.dat` per day covering every monitor worldwide
(~460 KB). We download each day, keep only the Jakarta rows, and save them
untouched to data/raw/airnow_jakarta_daily.dat. No API key needed.
Source: https://files.airnowtech.org/airnow/YYYY/YYYYMMDD/daily_data.dat
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "airnow_jakarta_daily.dat"
URL = "https://files.airnowtech.org/airnow/{d:%Y}/{d:%Y%m%d}/daily_data.dat"


def jakarta_rows(day):
    for attempt in range(3):
        try:
            r = requests.get(URL.format(d=day), timeout=60)
            if r.status_code == 200:
                return [l for l in r.text.splitlines() if "Jakarta" in l]
            print(day.date(), "HTTP", r.status_code)  # day missing from the archive
            return []
        except requests.RequestException:
            time.sleep(5)
    print("gave up on", day.date())
    return []


done = set()
if OUT.exists():  # resume: skip days already saved
    done = {line.split("|")[0] for line in OUT.read_text().splitlines()}
todo = [d for d in pd.date_range("2024-01-01", "2025-12-31") if d.strftime("%m/%d/%y") not in done]

# 8 downloads at a time: one-by-one takes ~1 hour for 731 files.
with OUT.open("a") as f, ThreadPoolExecutor(8) as pool:
    for day, rows in zip(todo, pool.map(jakarta_rows, todo)):
        f.write("".join(l + "\n" for l in rows))
        f.flush()
        print(day.date(), len(rows), "rows", flush=True)

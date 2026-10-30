# Jakarta PM2.5: 2024 heatmap + predicting 2025

Live pages: [2024 heatmap](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/heatmap_2024.html) ·
[2025 official vs predicted](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/heatmap_2025_compare.html) ·
[model comparison chart](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/model_comparison.png)

I built [Breathability](https://github.com/SycthePlays/Breathability) with a team at a hackathon.
It's an app that picks the route through Jakarta with the cleanest air. This repo is me taking one
piece of that further on my own. The question is simple: if a model only learns from Jakarta's air
from 2020 to 2024, how close can it get to what the official monitors actually measured in 2025?

There are two heatmaps with a day slider, where green means clean air and red means bad air. The
first shows 2024. The second puts the official 2025 readings next to what the model predicted, so
you can slide through the year and see where it gets things wrong. The scores for every model are
in `outputs/results.csv` and in the chart.

## Results

Every model here was trained on 2020–2024 (about 3,200 station-days) and tested on the 2025 days
that have data.

| Task | Model | MAE (µg/m³) | RMSE | R² | AQI category correct |
|---|---|---|---|---|---|
| A: weather only | Climatology (baseline) | 10.0 | 12.4 | 0.19 | 67% |
| | Ridge (linear) | 11.0 | 12.9 | 0.12 | 60% |
| | Random Forest | 10.1 | 12.1 | 0.23 | 61% |
| | LightGBM | 10.0 | 12.1 | 0.23 | 63% |
| B: next day | Climatology (baseline) | 10.0 | 12.4 | 0.19 | 67% |
| | Persistence, "tomorrow = today" (baseline) | 7.6 | 10.1 | 0.46 | 70% |
| | Ridge (linear) | 7.8 | 9.5 | 0.52 | 70% |
| | Random Forest | 7.1 | 8.9 | 0.58 | 69% |
| | LightGBM | 7.2 | 9.0 | 0.58 | 73% |

The weather-only models did no better than just guessing the 2020–2024 average for that month.
LightGBM, the best of them, tied that guess (both 10.0 µg/m³), and it got the AQI category right
less often (63% vs 67%). Ridge was clearly worse. Going from one year of training to five helped
the simple average more than the models: when I trained on 2024 alone, the average's error was
10.8 and LightGBM's was 10.3. I think the reason is that the weather data comes from a grid about
25 km wide, and that's way too coarse to explain what's happening on a Jakarta street.

I also tried going back further, training on 2016–2024 (about 5,900 station-days, since the
monitors start in 2016). It made things worse: the monthly average's error went from 10.0 to 11.0
and weather-only LightGBM's from 10.0 to 10.3. Jakarta's air was dirtier back then (yearly averages
of 41, 30, 41 and 46 µg/m³ for 2016–2019, against 29 in 2025), so the models learned to expect
around 38 and overshot 2025 by about 8. More data only helps if the past looks like the future.
That's why the results above use 2020–2024.

Knowing yesterday's reading helped a lot more than any model did. Just saying "tomorrow will be
the same as today" cut the error by about a quarter (10.0 down to 7.6). Random Forest and LightGBM
squeezed out a bit more (7.1 and 7.2), and LightGBM got the AQI category right most often of
anything (73%). Ridge did slightly worse than the simple guess.

I also tracked "within one AQI category", but every model scores 99% or more on it, baselines
included. Almost every day in Jakarta is either Moderate or Unhealthy for Sensitive Groups, so
being one category off is hard to avoid being right about.

The errors also jump around a lot by month (`outputs/monthly_mae.csv`). For the weather-only
models they go from about 8 µg/m³ in January and May 2025 up to 13 to 15 in April.

## The two tasks

Task A is the main one: predict PM2.5 from the weather. The model gets daily temperature,
humidity, rain from that day and the day before, air pressure, wind speed and direction, day of
the week, time of year, and which station it is. It never sees any past PM2.5. It does use the
real 2025 weather, though, so this isn't a true forecast. It's more like "given this weather,
what should the air have been?"

Task B predicts the next day. It gets everything from Task A plus the measured PM2.5 from the
two days before.

## How I did it

The thing being predicted is the daily average PM2.5 from the two US Embassy monitors, Jakarta
Central and Jakarta South, which I downloaded from the AirNow archive. I dropped 9 days where
Jakarta South read zero, below zero or almost zero (−3.0 to 2.0 µg/m³, in Sep 2020, Oct 2023,
Oct 2024 and Oct 2025). Jakarta's air is never that clean, and on the 2024 dates Jakarta Central
read 23 to 45 with no rain, so those were sensor faults.

I split by time: train on 2020–2024, test on 2025. If you shuffle the days randomly instead, the model
gets to peek at the future and the scores look better than they really are. I picked the settings
for each model using cross-validation inside 2020–2024 only (4 folds, each one training on
earlier data and checking on what came after), so 2025 stayed untouched until the final test.

To turn PM2.5 into AQI I used the US EPA breakpoints from the May 2024 update, where "Good" now
ends at 9.0 µg/m³. Those breakpoints are meant for 24-hour averages, which is exactly what I have.
The full table is in `src/aqi.py`.

For the heatmap, I split the area into squares about 550 m wide. Each one is a weighted average of
the stations that reported that day, with closer stations counting more. The colours are drawn
as a see-through layer over a street map so you can tell where things are, and the map blends
neighbouring squares into a smooth gradient. The colours go from green to red across the
lowest and highest AQI on that page (16 to 170 for 2024, 21 to 166 for 2025). That makes small
differences easier to see, but it means the colours don't match the official AQI band colours,
so check the colour bar for the actual number. With only two stations, almost all of the
map is an estimate. On days when only one station reported, the whole map is just that one value.
The grey dots are the districts from Breathability (`Breathability_v2/lib/jakartaAreas.js`), and
you can hover over them to see the estimate for each one.

## Limitations

Only 2 stations had usable data for 2020–2025. Jakarta Central goes quiet in April 2025, and
both stop in early November 2025, which leaves 388 station-days to test on. Open-Meteo also has no
boundary-layer height for January to June 2024. That's a big deal for pollution, since it's
basically how high up the dirty air can spread, but Ridge and Random Forest can't handle
missing values, so I left it out. The two stations are also close enough that they fall in nearly the same weather
grid square, so they get almost the same weather. The data sources I tried that didn't work out
are in `data/SOURCES.md`.

## Run it

```bash
pip install -r requirements.txt
cd src
python fetch_airnow.py      # a few minutes, downloads 731 daily AirNow files, keeps Jakarta rows
python build_dataset.py     # weather + cleaning -> data/processed/dataset.parquet
python train_eval.py        # models + outputs/results.csv + chart
python make_heatmap.py      # needs ../Breathability_v2 next to this repo for the district dots
python aqi.py               # self-check of the AQI table
```

`forecast.py` is an older experiment (hourly, 1 to 24 hours ahead, one station). It isn't part of
this pipeline.

## Data credits

PM2.5 comes from the U.S. Department of State monitors in Jakarta, through AirNow
(files.airnowtech.org). Weather comes from the [Open-Meteo](https://open-meteo.com/) Historical
Weather API, CC BY 4.0, which is built on ERA5 reanalysis from the Copernicus Climate Change
Service. The street map under the heatmaps is © OpenStreetMap contributors, styled by CARTO.

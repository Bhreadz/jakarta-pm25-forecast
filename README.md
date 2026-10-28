# Jakarta PM2.5: 2024 heatmap + predicting 2025

Live pages: [2024 heatmap](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/heatmap_2024.html) ·
[2025 official vs predicted](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/heatmap_2025_compare.html) ·
[model comparison chart](https://bhreadz.github.io/jakarta-pm25-forecast/outputs/model_comparison.png)

I built [Breathability](https://github.com/SycthePlays/Breathability) with a team at a hackathon.
It's an app that picks the route through Jakarta with the cleanest air. This repo is me taking one
piece of that further on my own. The question is simple: if a model only learns from Jakarta's air
in 2024, how close can it get to what the official monitors actually measured in 2025?

There are two heatmaps with a day slider, where green means clean air and red means bad air. The
first shows 2024. The second puts the official 2025 readings next to what the model predicted, so
you can slide through the year and see where it gets things wrong. The scores for every model are
in `outputs/results.csv` and in the chart.

## Results

Every model here was trained on 2024 and tested on the 2025 days that have data.

| Task | Model | MAE (µg/m³) | RMSE | R² | AQI category correct |
|---|---|---|---|---|---|
| A: weather only | Climatology (baseline) | 10.8 | 13.5 | 0.05 | 66% |
| | Ridge (linear) | 10.6 | 13.3 | 0.08 | 62% |
| | Random Forest | 10.7 | 13.4 | 0.08 | 63% |
| | LightGBM | 10.5 | 13.3 | 0.09 | 63% |
| B: next day | Climatology (baseline) | 10.8 | 13.6 | 0.04 | 65% |
| | Persistence, "tomorrow = today" (baseline) | 7.7 | 10.2 | 0.46 | 70% |
| | Ridge (linear) | 8.0 | 10.1 | 0.48 | 70% |
| | Random Forest | 7.3 | 9.4 | 0.55 | 70% |
| | LightGBM | 8.5 | 10.8 | 0.40 | 65% |

The weather-only models barely did better than just guessing last year's average for that month.
LightGBM, the best of them, was 0.3 µg/m³ closer than that guess, and it actually got the AQI
category right less often. I think the reason is that the weather data comes from a grid about
25 km wide, and that's way too coarse to explain what's happening on a Jakarta street.

Knowing yesterday's reading helped a lot more than any model did. Just saying "tomorrow will be
the same as today" cut the error by about 30%. Random Forest squeezed out a bit more (7.7 down to
7.3). LightGBM ended up worse than that simple guess.

I also tracked "within one AQI category", but every model scores about 99% on it, baselines
included. Almost every day in Jakarta is either Moderate or Unhealthy for Sensitive Groups, so
being one category off is hard to avoid being right about.

The errors also jump around a lot by month (`outputs/monthly_mae.csv`). For the weather-only
models they go from about 6 to 8 µg/m³ in January and September 2025 up to about 16 in October.

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
Central and Jakarta South, which I downloaded from the AirNow archive. 

I split by time: train on 2024, test on 2025. If you shuffle the days randomly instead, the model
gets to peek at the future and the scores look better than they really are. I picked the settings
for each model using cross-validation inside 2024 only (4 folds, each one training on earlier
months and checking on later ones), so 2025 stayed untouched until the final test.

To turn PM2.5 into AQI I used the US EPA breakpoints from the May 2024 update, where "Good" now
ends at 9.0 µg/m³. Those breakpoints are meant for 24-hour averages, which is exactly what I have.
The full table is in `src/aqi.py`.

For the heatmap, each grid square (about 1 km) is a weighted average of the stations that
reported that day, with closer stations counting more. With only two stations, almost all of the
map is an estimate. On days when only one station reported, the whole map is just that one value.
The grey dots are the districts from Breathability (`Breathability_v2/lib/jakartaAreas.js`), and
you can hover over them to see the estimate for each one.

## Limitations

Only 2 stations had usable data for both years. Jakarta Central goes quiet in April 2025, and
both stop in early November 2025, which leaves 391 station-days to test on. Open-Meteo also has no
boundary-layer height for January to June 2024. That's a big deal for pollution, since it's
basically how high up the dirty air can spread, but with half the training year missing I had to
leave it out. The two stations are also close enough that they fall in nearly the same weather
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
Service.

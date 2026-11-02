"""Open-Meteo weather: reanalysis to train on, archived forecasts to test with.

Legality at τ (09:30 on D-1), source by source:
- ERA5 reanalysis is the weather that happened, published about 5 days late. Legal at τ
  for D-7 (last week's weather), a leak for D. Used for training rows and for D-7.
- Previous Runs `_previous_day2` is the value forecast 48 hours before each valid hour.
  The last hour of D was forecast on D-1 before 01:00, so every hour of D is legal at τ.
  Archived from January 2024, so it covers the 2024 validation year and the 2025 test year.
- Rejected: `_previous_day1` (forecast 24 h before valid time) is a leak for every hour of
  D after about 09:00, because those forecasts were made after τ.

The cost of day2: forecasts about 24 hours staler than NESO's. A known handicap, stated.
"""
import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
REANALYSIS_MODEL = "era5"
FORECAST_MODEL = "ecmwf_ifs025"  # ECMWF open data; archived since January 2024
FORECAST_LEAD_DAYS = 2

REANALYSIS_DELAY = pd.Timedelta(days=5)
# A forecast "made 48 h before valid time" could mean the model run started then; its
# output appears a few hours later. 8 hours is a conservative allowance for that.
FORECAST_PUBLISH_LAG = pd.Timedelta(hours=8)

# Open-Meteo variable -> short name used in the features
VARIABLES = {"temperature_2m": "temp", "shortwave_radiation": "ghi", "wind_speed_10m": "wind", "cloud_cover": "cloud"}
HOUR_MEAN = {"ghi"}  # averaged over the preceding hour; the rest are instantaneous

# (latitude, longitude, population weight, solar weight). Rough, and tunable on 2024 only.
# Temperature and wind follow where people live; irradiance and cloud follow where most
# embedded solar is (south and south-west England, East Anglia).
SITES = {
    "london":      (51.51, -0.13, 9.5, 1.0),
    "birmingham":  (52.48, -1.90, 2.9, 1.5),
    "manchester":  (53.48, -2.24, 2.9, 0.5),
    "leeds":       (53.80, -1.55, 2.4, 0.5),
    "glasgow":     (55.86, -4.25, 1.8, 0.2),
    "bristol":     (51.45, -2.59, 1.0, 2.5),
    "southampton": (50.90, -1.40, 1.0, 2.0),
    "norwich":     (52.63, 1.30, 0.8, 1.5),
}
WEIGHTING = {"temp": "people", "wind": "people", "ghi": "solar", "cloud": "solar"}


def _request(url, params):
    response = requests.get(url, params=params, timeout=120)
    if response.status_code == 400:
        raise ValueError(f"Open-Meteo rejected the request: {response.json().get('reason')}")
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def fetch_sites(kind, start, end):
    """Hourly values for every site, long format [time_utc, site, variable, value].

    kind: "reanalysis" (ERA5) or "forecast" (Previous Runs, 48 h before valid time).
    """
    if kind == "reanalysis":
        url, model, suffix = ARCHIVE_URL, REANALYSIS_MODEL, ""
    elif kind == "forecast":
        url, model, suffix = PREVIOUS_RUNS_URL, FORECAST_MODEL, f"_previous_day{FORECAST_LEAD_DAYS}"
    else:
        raise ValueError(f"unknown weather kind {kind!r}")
    names = list(SITES)
    requested = {f"{v}{suffix}": short for v, short in VARIABLES.items()}
    params = {
        "latitude": ",".join(str(SITES[s][0]) for s in names),
        "longitude": ",".join(str(SITES[s][1]) for s in names),
        "hourly": ",".join(requested),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": model,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    frames = []
    for name, block in zip(names, _request(url, params)):
        hourly = block["hourly"]
        times = pd.to_datetime(hourly["time"]).tz_localize("UTC")
        for api_name, short in requested.items():
            frames.append(pd.DataFrame({"time_utc": times, "site": name, "variable": short,
                                        "value": pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce")}))
    return pd.concat(frames, ignore_index=True)


def national(long):
    """Site values -> one weighted national series per variable, indexed by hour (UTC).

    Strict: an hour with any site missing is NaN, so a gap never silently reweights the sites.
    """
    wide = long.pivot_table(index="time_utc", columns=["variable", "site"], values="value", aggfunc="first", dropna=False)
    out = {}
    for short, scheme in WEIGHTING.items():
        col = 2 if scheme == "people" else 3
        w = pd.Series({s: SITES[s][col] for s in SITES})
        block = wide[short].reindex(columns=w.index)
        out[short] = (block * w).sum(axis=1, min_count=len(w)) / w.sum()
    hourly = pd.DataFrame(out)
    hourly.index = pd.DatetimeIndex(hourly.index).as_unit("ns")
    return hourly.sort_index()


def _interp(series, targets):
    """Linear interpolation between neighbouring points at most 1 hour apart; NaN otherwise."""
    s = series[~series.index.duplicated()].sort_index()
    t = s.index.as_unit("ns").asi8
    v = s.to_numpy(dtype=float)
    x = pd.DatetimeIndex(targets).as_unit("ns").asi8
    out = np.full(len(x), np.nan)
    if len(t) < 2:
        return out
    left = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
    right = left + 1
    step = t[right] - t[left]
    inside = (x >= t[left]) & (x <= t[right]) & (step <= 3_600 * 10**9)
    frac = (x - t[left]) / step
    est = v[left] * (1 - frac) + v[right] * frac
    out[inside] = est[inside]
    return out


def to_half_hourly(hourly, sp_starts):
    """Hourly national weather -> one value per settlement period, at the SP's midpoint.

    Instantaneous variables sit on the hour. Hour-mean variables (irradiance) describe the
    hour BEFORE their timestamp, so they are moved back 30 minutes to the hour's centre
    before interpolating.
    """
    sp_starts = pd.DatetimeIndex(sp_starts).as_unit("ns")
    midpoints = sp_starts + pd.Timedelta(minutes=15)
    out = {}
    for col in hourly.columns:
        s = hourly[col]
        if col in HOUR_MEAN:
            s = s.set_axis(s.index - pd.Timedelta(minutes=30))
        out[col] = _interp(s, midpoints)
    return pd.DataFrame(out, index=sp_starts)


def info_time(kind, sp_starts):
    """When each half-hour's value became public (the latest of the two hours interpolated)."""
    latest_hour = pd.DatetimeIndex(sp_starts).as_unit("ns") + pd.Timedelta(hours=1)
    if kind == "reanalysis":
        return latest_hour + REANALYSIS_DELAY
    return latest_hour - pd.Timedelta(days=FORECAST_LEAD_DAYS) + FORECAST_PUBLISH_LAG

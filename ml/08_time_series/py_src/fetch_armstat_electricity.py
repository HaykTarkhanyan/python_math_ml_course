"""Fetch the monthly Armenian electricity/gas/steam production series from ArmStatBank.

Source: Statistical Committee of the Republic of Armenia (armstat), table
IC-ind-m-01.px -- "Volume of Industrial Production at current prices by types of
economic activity according to the two-digit classification, years and by months",
category 31 = "35. Electricity, gas, steam and air conditioning supply".

Portal: https://statbank.armstat.am  (PX-Web; the click-through UI at
armstat.am/en/?nid=17 is backed by the JSON API used here).

Writes ml/08_time_series/data/armstat_electricity_monthly.csv with columns:
    date        -- YYYY-MM-01, month start
    value_kdram -- volume of production, thousand drams, CURRENT prices (nominal)

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/08_time_series/py_src/fetch_armstat_electricity.py

Conventions (repo CLAUDE.md): console + logs/ logging, f-strings, fail loud.

NOTE ON TLS: statbank.armstat.am serves a certificate whose common name does not
match the host (net::ERR_CERT_COMMON_NAME_INVALID), so certificate verification is
explicitly disabled below. This is a deliberate, logged decision for one specific
public-statistics host, NOT a silent fallback -- the script warns every run.
"""

import json
import logging
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "armstat_electricity_monthly.csv"

BASE = "https://statbank.armstat.am/api/v1/en/ArmStatBank"
TABLE = (
    f"{BASE}/3%20Industry,%20Construction,%20trade%20and%20services/32%20Industry/"
    "321%20Industry,%20RA/3212%20Monthly%20indicators/IC-ind-m-01.px"
)
ACTIVITY_VAR = "types of economic activity according to the two-digit classification"
ELECTRICITY_CODE = "31"  # "35. Electricity, gas, steam and air conditioning supply"

# The series is nominal. Anything below this is almost certainly a reporting artefact
# rather than a real month of national electricity production.
MIN_PLAUSIBLE_KDRAM = 1_000_000  # 1 billion drams


def build_logger() -> logging.Logger:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "fetch_armstat_electricity.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def fetch(log: logging.Logger) -> dict:
    log.warning(
        "TLS verification DISABLED for statbank.armstat.am: the host serves a cert with a "
        "mismatched common name. Deliberate and scoped to this public-statistics host."
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    query = {
        "query": [
            {"code": ACTIVITY_VAR, "selection": {"filter": "item", "values": [ELECTRICITY_CODE]}},
            {"code": "years", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "by months", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    request = urllib.request.Request(
        TABLE, data=json.dumps(query).encode(), headers={"Content-Type": "application/json"}
    )
    log.info(f"POST {TABLE.split('/')[-1]} activity={ELECTRICITY_CODE}")
    with urllib.request.urlopen(request, context=ctx, timeout=120) as response:
        return json.load(response)


def to_frame(payload: dict, log: logging.Logger) -> pd.DataFrame:
    label = payload["dimension"][ACTIVITY_VAR]["category"]["label"]
    log.info(f"series: {list(label.values())[0]}")

    years = [int(y) for y in payload["dimension"]["years"]["category"]["label"].values()]
    values = payload["value"]
    expected = len(years) * 12
    if len(values) != expected:
        raise RuntimeError(f"expected {expected} cells for {len(years)} years, got {len(values)}")

    rows = []
    for year_index, year in enumerate(years):
        for month in range(12):
            value = values[year_index * 12 + month]
            if value is None:
                continue  # not yet published; the series simply ends mid-year
            rows.append({"date": pd.Timestamp(year=year, month=month + 1, day=1),
                         "value_kdram": float(value)})

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if frame.empty:
        raise RuntimeError("armstat returned no non-null observations for this series")

    gaps = frame["date"].diff().dt.days.dropna()
    if (gaps > 32).any():
        bad = frame.loc[1:][gaps.values > 32, "date"].tolist()
        raise RuntimeError(f"series is not contiguous monthly; gap before {bad}")

    too_small = frame[frame["value_kdram"] < MIN_PLAUSIBLE_KDRAM]
    if not too_small.empty:
        raise RuntimeError(
            f"{len(too_small)} month(s) below {MIN_PLAUSIBLE_KDRAM:,} kdram, first "
            f"{too_small.iloc[0]['date'].date()} = {too_small.iloc[0]['value_kdram']:,.0f}. "
            "Suspect a units change or a reporting artefact; inspect before using."
        )
    return frame


def main() -> None:
    log = build_logger()
    frame = to_frame(fetch(log), log)

    OUT_CSV.parent.mkdir(exist_ok=True)
    frame.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")

    first, last = frame.iloc[0], frame.iloc[-1]
    log.info(f"n={len(frame)} months, {first['date'].date()} .. {last['date'].date()}")
    log.info(
        f"level: min {frame['value_kdram'].min() / 1e6:.1f}B, "
        f"max {frame['value_kdram'].max() / 1e6:.1f}B drams"
    )
    log.info(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

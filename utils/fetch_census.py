"""Download the real-world inputs from the U.S. Census Bureau API into data/.

Usage (from the project root):
    python -m utils.fetch_census

Needs a free API key (https://api.census.gov/data/key_signup.html), supplied either as the
CENSUS_API_KEY environment variable or in a ``.env`` file containing ``CENSUS_API_KEY=...``.
``.env`` is git-ignored, so the key is never committed.

Outputs
  data/initial_state_<year>.csv    state, population, median_rent_annual, median_income
  data/observed_net_migration.csv  state, period, net_domestic_migration   (used only to check
                                   the baseline model, never to set it up)
"""
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

START_YEAR = 2010  # ACS 1-year vintage used for initial conditions

# State FIPS codes. Chosen to contrast growth, rent level and housing supply (see proposal).
STATES = {
    "California": "06", "New York": "36", "Illinois": "17", "Ohio": "39",
    "Texas": "48", "Florida": "12", "Colorado": "08", "North Carolina": "37",
}

ACS_VARS = {
    "B01003_001E": "population",
    "B25064_001E": "median_rent_monthly",   # median gross rent, $ per month
    "B19013_001E": "median_income",          # median household income, $ per year
}


def get_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CENSUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RuntimeError("No Census API key found. Set CENSUS_API_KEY or add it to .env "
                           "(see the docstring in utils/fetch_census.py).")
    return key


def _query(url: str, params: dict, key: str) -> pd.DataFrame:
    r = requests.get(url, params={**params, "key": key}, timeout=60)
    r.raise_for_status()
    rows = r.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_initial_state(key: str, year: int = START_YEAR) -> pd.DataFrame:
    df = _query(f"https://api.census.gov/data/{year}/acs/acs1",
                {"get": "NAME," + ",".join(ACS_VARS), "for": "state:*"}, key)
    df = df[df["state"].isin(STATES.values())].rename(columns={"NAME": "state_name", **ACS_VARS})
    for col in ACS_VARS.values():
        df[col] = pd.to_numeric(df[col])
    df["median_rent_annual"] = df["median_rent_monthly"] * 12
    df["state"] = df["state_name"]
    order = {name: i for i, name in enumerate(STATES)}
    df = df.sort_values("state", key=lambda s: s.map(order))
    return df[["state", "population", "median_rent_annual", "median_income"]].reset_index(drop=True)


def fetch_observed_migration(key: str) -> pd.DataFrame:
    """Net domestic migration per state for each full July-June estimate year (2010-11 to 2018-19)."""
    df = _query("https://api.census.gov/data/2019/pep/components",
                {"get": "NAME,DOMESTICMIG,PERIOD_CODE,PERIOD_DESC", "for": "state:*"}, key)
    df = df[df["state"].isin(STATES.values())]
    df = df[df["PERIOD_CODE"].astype(int) >= 2]  # period 1 is a partial (Apr-Jun 2010) period
    out = pd.DataFrame({"state": df["NAME"], "period": df["PERIOD_DESC"],
                        "net_domestic_migration": pd.to_numeric(df["DOMESTICMIG"])})
    order = {name: i for i, name in enumerate(STATES)}
    return out.sort_values(["state", "period"],
                           key=lambda s: s.map(order) if s.name == "state" else s).reset_index(drop=True)


def main() -> None:
    key = get_api_key()
    DATA_DIR.mkdir(exist_ok=True)
    init = fetch_initial_state(key)
    mig = fetch_observed_migration(key)
    init.to_csv(DATA_DIR / f"initial_state_{START_YEAR}.csv", index=False)
    mig.to_csv(DATA_DIR / "observed_net_migration.csv", index=False)
    print(init.to_string(index=False))
    print(f"\nWrote {len(init)} states and {len(mig)} migration rows to {DATA_DIR}")


if __name__ == "__main__":
    main()

# Data loading helpers used across the notebooks.
# Kept here so we're not copy-pasting the same loading/aggregation code
# into every notebook.

import glob

import pandas as pd

# The 8 SA metros (2016/2021 demarcation). Metro vs local municipality
# is a useful split since turnout patterns differ between them.
METRO_CODES = ["BUF", "CPT", "EKU", "ETH", "JHB", "MAN", "NMA", "TSH"]


def load_province_files(year_dir: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Load and combine all the per-province CSVs for one election year.

    encoding: 2016 and 2021 files are UTF-8. The 2011 files are UTF-16,
    so pass encoding="utf-16" when loading that year.
    """
    files = sorted(glob.glob(f"{year_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {year_dir}")
    print(f"{year_dir}: found {len(files)} province files (encoding={encoding})")
    return pd.concat(
        [pd.read_csv(f, encoding=encoding) for f in files], ignore_index=True
    )


def to_ward_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse station x party rows down to one row per ward.

    Filters to the Ward ballot only (each voter casts one ward vote
    regardless of municipality type, so it's the most consistent turnout
    measure across metros and local municipalities).
    """
    df = df[df["BallotType"] == "Ward"].copy()

    station = df.groupby(
        ["Province", "Municipality", "Ward", "VotingDistrict"], as_index=False
    ).agg(
        RegisteredVoters=("RegisteredVoters", "first"),
        SpoiltVotes=("SpoiltVotes", "first"),
        TotalValidVotes=("TotalValidVotes", "sum"),
    )

    ward = station.groupby(["Province", "Municipality", "Ward"], as_index=False).agg(
        RegisteredVoters=("RegisteredVoters", "sum"),
        SpoiltVotes=("SpoiltVotes", "sum"),
        TotalValidVotes=("TotalValidVotes", "sum"),
    )
    ward["VotesCast"] = ward["TotalValidVotes"] + ward["SpoiltVotes"]
    ward["Turnout"] = ward["VotesCast"] / ward["RegisteredVoters"]
    ward["SpoiltRatio"] = ward["SpoiltVotes"] / ward["VotesCast"]
    return ward


def get_spoilt_ratios(year_dir: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Raw year folder -> [Province, Ward, SpoiltRatio]."""
    raw = load_province_files(year_dir, encoding=encoding)
    ward = to_ward_level(raw)
    return ward[["Province", "Ward", "SpoiltRatio"]]


def load_ward_panel(path: str = "../data/processed/ward_panel.csv") -> pd.DataFrame:
    """Load the cleaned ward-level panel."""
    return pd.read_csv(path)


def add_metro_flag(df: pd.DataFrame, municipality_code_col: str = "MunicipalityCode") -> pd.DataFrame:
    """Add an IsMetro column based on the municipality code."""
    df = df.copy()
    df["IsMetro"] = df[municipality_code_col].isin(METRO_CODES)
    return df

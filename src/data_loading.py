"""Shared raw-data loading utilities."""

import glob

import pandas as pd


def load_province_files(year_dir: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Load and concatenate every per-province CSV in a year's raw data folder.

    Parameters
    ----------
    year_dir : str
        Path to a folder containing one CSV per province (e.g. data/raw/LGE2016).
    encoding : str
        File encoding. The 2016 and 2021 IEC exports are UTF-8. The 2011
        export was found to be UTF-16 (little-endian, with a BOM) — pass
        encoding="utf-16" for that year.
    """
    files = sorted(glob.glob(f"{year_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {year_dir}")
    print(f"{year_dir}: found {len(files)} province files (encoding={encoding})")
    return pd.concat(
        [pd.read_csv(f, encoding=encoding) for f in files], ignore_index=True
    )
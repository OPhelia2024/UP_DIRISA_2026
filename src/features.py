# Feature engineering functions for the turnout model.
#
# Main idea: every feature here can be built from a single completed
# election on its own (a "snapshot"). That way the same function works
# for training (2016 features -> known 2021 turnout) and for the real
# forecast (2021 features -> unknown 2026 turnout), with no risk of
# accidentally using information that wouldn't exist yet at forecast time.

import numpy as np
import pandas as pd

from src.data_loading import add_metro_flag


def build_snapshot_features(panel: pd.DataFrame, spoilt_ratios: pd.DataFrame, year: int) -> pd.DataFrame:
    """Build snapshot features for one election year.

    panel: the ward_panel dataframe (has *_2016 / *_2021 columns)
    spoilt_ratios: [Province, Ward, SpoiltRatio] for this year
    year: 2016 or 2021
    """
    reg_col = f"RegisteredVoters_{year}"
    turnout_col = f"Turnout_{year}"

    df = panel[["Province", "Ward", "MunicipalityCode", reg_col, turnout_col]].copy()
    df = df.rename(columns={reg_col: "RegisteredVoters", turnout_col: "Turnout"})

    df = add_metro_flag(df)
    df["LogRegisteredVoters"] = np.log(df["RegisteredVoters"])

    # Municipality average turnout, leave-one-out (excludes the ward's own
    # turnout so a ward doesn't just see its own value reflected back).
    muni_sum = df.groupby("MunicipalityCode")["Turnout"].transform("sum")
    muni_count = df.groupby("MunicipalityCode")["Turnout"].transform("count")
    loo_avg = (muni_sum - df["Turnout"]) / (muni_count - 1)
    # Municipalities with only 1 ward have no other wards to average -
    # fall back to the ward's own turnout in that case.
    df["MunicipalityAvgTurnout"] = loo_avg.fillna(df["Turnout"])

    df = df.merge(spoilt_ratios, on=["Province", "Ward"], how="left")

    df = df.rename(columns={
        "Turnout": "Turnout_prior",
        "RegisteredVoters": "RegisteredVoters_prior",
        "LogRegisteredVoters": "LogRegisteredVoters_prior",
        "MunicipalityAvgTurnout": "MunicipalityAvgTurnout_prior",
        "SpoiltRatio": "SpoiltRatio_prior",
    })
    df["SnapshotYear"] = year
    return df


def build_trend_features(panel: pd.DataFrame, snapshot_features: pd.DataFrame, year: int, prior_year: int) -> pd.DataFrame:
    """Add turnout-change and registration-growth features, looking back
    from `prior_year` to `year` (both already completed elections).

    Same call shape works for training (2011->2016, predicting 2021) and
    for the real forecast (2016->2021, predicting 2026), so the deltas
    never touch the election being predicted.
    """
    delta_col = f"TurnoutDelta_{prior_year}_{year}"
    growth_col = f"RegistrationGrowth_{prior_year}_{year}"

    df = snapshot_features.copy()

    if delta_col in panel.columns and growth_col in panel.columns:
        trend = panel[["Province", "Ward", delta_col, growth_col]]
    else:
        trend = panel[["Province", "Ward", f"Turnout_{year}", f"Turnout_{prior_year}",
                        f"RegisteredVoters_{year}", f"RegisteredVoters_{prior_year}"]].copy()
        trend[delta_col] = trend[f"Turnout_{year}"] - trend[f"Turnout_{prior_year}"]
        trend[growth_col] = trend[f"RegisteredVoters_{year}"] / trend[f"RegisteredVoters_{prior_year}"] - 1
        trend = trend[["Province", "Ward", delta_col, growth_col]]

    df = df.merge(trend, on=["Province", "Ward"], how="left")
    df = df.rename(columns={delta_col: "TurnoutDelta_prior", growth_col: "RegistrationGrowth_prior"})
    return df


def prepare_model_inputs(df: pd.DataFrame, reference_columns=None):
    """Turn a feature table into model-ready X, plus the identifier columns
    kept aside for display.

    Drops Ward (just an ID, not a pattern) and SnapshotYear (same value for
    every row in a given table, so it's not useful). Also drops
    MunicipalityCode as a raw feature - with 200+ codes it's better to let
    IsMetro and MunicipalityAvgTurnout_prior carry that information, rather
    than encode the raw code directly, which can cause the model to
    memorise specific municipalities instead of learning a general pattern.
    Province (~9 categories) is one-hot encoded as-is.

    reference_columns: pass the training set's columns when encoding a
    different table (e.g. the forecast set), so both end up with the
    exact same columns in the same order.

    Returns (X, identifiers, groups) - identifiers is [Province, Ward,
    MunicipalityCode] for display, groups is MunicipalityCode for use
    with a grouped train/test split.
    """
    identifiers = df[["Province", "Ward", "MunicipalityCode"]].copy()
    groups = df["MunicipalityCode"]

    X = df.drop(columns=["Ward", "SnapshotYear", "MunicipalityCode"], errors="ignore")
    if "Target_Turnout" in X.columns:
        X = X.drop(columns=["Target_Turnout"])
    X = pd.get_dummies(X, columns=["Province"], drop_first=True)

    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)

    return X, identifiers, groups

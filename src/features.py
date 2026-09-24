"""Feature engineering functions for the turnout-prediction model.

Design principle (important — see 04_feature_engineering.ipynb for the full
explanation): every feature here is a *snapshot* feature, computable from a
single completed election alone. The same function is applied to the 2016
panel (to train against the known 2021 outcome) and to the 2021 panel (to
forecast the unknown 2026 outcome). Using one function for both is what
guarantees the feature set can't accidentally depend on data that doesn't
exist yet at forecast time — that consistency *is* the leakage guardrail.
"""

import numpy as np
import pandas as pd

from src.data_loading import add_metro_flag


def build_snapshot_features(panel: pd.DataFrame, spoilt_ratios: pd.DataFrame, year: int) -> pd.DataFrame:
    """Build snapshot features for a single election year.

    Parameters
    ----------
    panel : the ward_panel.csv DataFrame (has *_2016 and *_2021 columns)
    spoilt_ratios : DataFrame with [Province, Ward, SpoiltRatio] for this `year`,
        e.g. from src.data_loading.get_spoilt_ratios("../data/raw/LGE2016")
    year : 2016 or 2021 — which election's snapshot to build features from

    Returns
    -------
    One row per ward, with columns suffixed `_prior` to make clear these are
    inputs describing the *prior* election, not the outcome being predicted.
    """
    reg_col = f"RegisteredVoters_{year}"
    turnout_col = f"Turnout_{year}"

    df = panel[["Province", "Ward", "MunicipalityCode", reg_col, turnout_col]].copy()
    df = df.rename(columns={reg_col: "RegisteredVoters", turnout_col: "Turnout"})

    df = add_metro_flag(df)
    df["LogRegisteredVoters"] = np.log(df["RegisteredVoters"])

    # Municipality-average turnout, LEAVE-ONE-OUT (excludes the ward's own
    # turnout from its municipality's average) so a ward's feature never
    # trivially encodes its own target-adjacent value.
    muni_sum = df.groupby("MunicipalityCode")["Turnout"].transform("sum")
    muni_count = df.groupby("MunicipalityCode")["Turnout"].transform("count")
    loo_avg = (muni_sum - df["Turnout"]) / (muni_count - 1)
    # Municipalities with only 1 ward have no "other wards" to average —
    # fall back to the ward's own turnout in that edge case.
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
    """Attach lookback trend features (from `prior_year` -> `year`) to an
    existing snapshot feature table.

    Both `year` and `prior_year` must already be completed elections — this
    is what keeps the trend features leakage-safe. E.g. year=2016,
    prior_year=2011 is safe to use when predicting Turnout_2021, because
    neither 2016 nor 2011 touches 2021 anywhere in their calculation. The
    exact same call shape (year=2021, prior_year=2016) is then safe when
    forecasting the unknown Turnout_2026.

    If the panel already has precomputed Turnout/RegistrationGrowth delta
    columns for this exact (prior_year, year) pair (as ward_panel_3yr.csv
    does for 2011->2016), those are reused directly. Otherwise they are
    computed here from the raw Turnout/RegisteredVoters columns — this is
    what lets the *same* function serve both the training pair (which
    arrives with deltas precomputed) and the forecast pair (2016->2021,
    which does not, since nobody had precomputed a lookback for it).
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

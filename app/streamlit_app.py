"""
Streamlit app for the 2026 LGE ward turnout forecast.

Model is trained on 2016 snapshot features -> known 2021 turnout
(train_features.csv), then applied to 2021 snapshot features
(forecast_features.csv) to actually forecast 2026. See notebooks 03-06
for the full EDA/feature engineering/modeling writeup.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import prepare_model_inputs

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

st.set_page_config(page_title="2026 LGE Turnout Forecast", layout="wide")


@st.cache_data
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    forecast_df = pd.read_csv(DATA_DIR / "forecast_features.csv")
    return train_df, forecast_df


@st.cache_resource
def train_model(train_df: pd.DataFrame):
    """Train the model, and separately measure its accuracy on a held-out
    set of whole municipalities (not a random row split - see
    05_modeling.ipynb for why a random split gives an inflated score here).
    """
    X, _, groups = prepare_model_inputs(train_df)
    y = train_df["Target_Turnout"]

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    eval_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    eval_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = eval_model.predict(X.iloc[test_idx])
    r2 = r2_score(y.iloc[test_idx], preds)
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))

    # Deployed model: refit on all training data for the best forecast
    final_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    final_model.fit(X, y)

    return final_model, list(X.columns), r2, rmse


def main():
    st.title("2026 Local Government Election - Ward Turnout Forecast")
    st.caption(
        "Predicts expected voter turnout per ward for the 4 November 2026 LGE, "
        "using turnout and registration patterns from the 2016 and 2021 elections."
    )

    train_df, forecast_df = load_data()
    model, feature_columns, r2, rmse = train_model(train_df)

    with st.sidebar:
        st.header("Model performance")
        st.metric("R² (held-out municipalities)", f"{r2:.3f}")
        st.metric("Typical error (RMSE)", f"{rmse * 100:.1f}%")
        st.caption(
            "Evaluated on wards from municipalities not seen during training "
            "(a grouped split) rather than a plain random split - this avoids "
            "the model just learning municipality-specific quirks."
        )
        st.markdown("---")
        st.markdown(
            "**Known limitations:**\n"
            "- Cannot see unobserved factors: weather, local issues, candidate scandals.\n"
            "- Trained on only two prior elections (2016, 2021).\n"
            "- Predictions are estimates, not guarantees - use as a planning signal, not a certainty."
        )

    st.subheader("Forecast a ward's 2026 turnout")

    col1, col2, col3 = st.columns(3)
    with col1:
        province = st.selectbox("Province", sorted(forecast_df["Province"].unique()))
    province_wards = forecast_df[forecast_df["Province"] == province]

    with col2:
        municipality = st.selectbox(
            "Municipality code", sorted(province_wards["MunicipalityCode"].unique())
        )
    muni_wards = province_wards[province_wards["MunicipalityCode"] == municipality]

    with col3:
        ward = st.selectbox("Ward", sorted(muni_wards["Ward"].unique()))

    selected = muni_wards[muni_wards["Ward"] == ward].iloc[[0]]

    if st.button("Predict 2026 turnout", type="primary"):
        X_selected, _, _ = prepare_model_inputs(selected, reference_columns=feature_columns)
        prediction = model.predict(X_selected)[0]

        st.markdown("### Result")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Predicted 2026 turnout", f"{prediction * 100:.1f}%")
        pcol2.metric(
            "2021 turnout (for comparison)",
            f"{selected['Turnout_prior'].values[0] * 100:.1f}%",
            delta=f"{(prediction - selected['Turnout_prior'].values[0]) * 100:.1f} %",
        )
        st.caption(
            f"Typical error for this kind of prediction is about ±{rmse * 100:.1f} "
            "%, based on held-out validation."
        )

    st.markdown("---")
    st.subheader("At-risk wards: lowest predicted 2026 turnout")
    st.caption("Useful for flagging wards where outreach or logistics planning may matter most.")

    X_all, ids_all, _ = prepare_model_inputs(forecast_df, reference_columns=feature_columns)
    all_preds = model.predict(X_all)
    results = ids_all.copy()
    results["Predicted_2026_Turnout"] = all_preds
    results["Turnout_2021"] = forecast_df["Turnout_prior"]
    lowest = results.sort_values("Predicted_2026_Turnout").head(15)
    lowest_display = lowest.copy()
    lowest_display["Predicted_2026_Turnout"] = (lowest_display["Predicted_2026_Turnout"] * 100).round(1)
    lowest_display["Turnout_2021"] = (lowest_display["Turnout_2021"] * 100).round(1)
    st.dataframe(lowest_display, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

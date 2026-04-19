import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from cleaning_utils import clean_dataset, transform_features_for_prediction

st.set_page_config(page_title="Predict Startup Success", layout="wide")

st.title("Predict Startup Success")
st.write(
    "Upload a CSV file. The app will clean the data, transform it using the same preprocessing logic used during training, "
    "align it to the training features, scale the required numeric columns, and then make predictions."
)


# LOAD SAVED ARTIFACTS
ARTIFACT_DIR = "saved_artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
NUMERIC_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "numeric_columns.pkl")

missing_artifacts = [
    path for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, NUMERIC_COLUMNS_PATH]
    if not os.path.exists(path)
]

if missing_artifacts:
    st.warning(
        "Saved model artifacts were not found. Please go to the Model Testing page first so the best model artifacts can be created."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
numeric_cols = joblib.load(NUMERIC_COLUMNS_PATH)

st.success("Saved model artifacts loaded successfully.")
st.info("Predictions are made using the saved best model (XGBoost).")


# HELPER FUNCTION
def preprocess_uploaded_data_for_prediction(raw_df, feature_columns, scaler, numeric_cols):
    clean_results = clean_dataset(raw_df)
    df_clean = clean_results["df_clean"]

    df_model = transform_features_for_prediction(df_clean)

    df_model = df_model.reindex(columns=feature_columns, fill_value=0)
    df_model = df_model.astype(float)

    df_scaled = df_model.copy()
    existing_numeric_cols = [col for col in numeric_cols if col in df_scaled.columns]

    if existing_numeric_cols:
        df_scaled[existing_numeric_cols] = scaler.transform(df_scaled[existing_numeric_cols])

    return df_scaled, df_clean, df_model



# OPTIONAL TEMPLATE INFO
with st.expander("Expected raw input structure", expanded=False):
    st.write("Your CSV should contain the raw startup columns before encoding, not the final training columns.")
    st.write("Examples of useful raw columns include:")
    st.write([
        "state_code",
        "category_code",
        "latitude",
        "longitude",
        "age_first_funding_year",
        "age_last_funding_year",
        "age_first_milestone_year",
        "age_last_milestone_year",
        "relationships",
        "funding_rounds",
        "funding_total_usd",
        "milestones",
        "avg_participants",
        "founded_at",
        "first_funding_at",
        "last_funding_at",
        "city",
        "labels",
        "status"
    ])
    st.write("If some irrelevant columns are included, the cleaning step will drop them.")
    st.write("If status is included, it will be ignored for prediction.")


# CSV UPLOAD
uploaded_file = st.file_uploader("Upload startup data as CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.markdown("### Uploaded Data Preview")
        st.dataframe(raw_df.head(), width="stretch")

        processed_df, df_clean, df_model_before_scaling = preprocess_uploaded_data_for_prediction(
            raw_df,
            feature_columns,
            scaler,
            numeric_cols
        )

        predictions = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)[:, 1]

        result_df = raw_df.copy()
        result_df["Predicted_Status"] = np.where(predictions == 1, "Acquired", "Closed")
        result_df["Success_Probability"] = probabilities

        st.markdown("### Prediction Results")
        st.dataframe(result_df, width="stretch")

        with st.expander("View cleaned data", expanded=False):
            st.dataframe(df_clean.head(), width="stretch")

        with st.expander("View transformed features before scaling", expanded=False):
            st.dataframe(df_model_before_scaling.head(), width="stretch")

        with st.expander("View final processed features used for prediction", expanded=False):
            st.dataframe(processed_df.head(), width="stretch")

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction results as CSV",
            data=csv,
            file_name="startup_predictions.csv",
            mime="text/csv"
        )


        # LOCAL SHAP EXPLANATION
        st.markdown("---")
        st.markdown("### Explain a Prediction")

        row_index = st.selectbox(
            "Select a row to explain",
            options=result_df.index.tolist()
        )

        st.write(f"Predicted Status: {result_df.loc[row_index, 'Predicted_Status']}")
        st.write(f"Success Probability: {result_df.loc[row_index, 'Success_Probability']:.4f}")

        explainer = shap.TreeExplainer(model)
        single_row = processed_df.iloc[[row_index]]
        shap_values_single = explainer.shap_values(single_row)

        # Handle binary-output shape cleanly
        if isinstance(shap_values_single, list):
            row_shap = shap_values_single[1][0]
            base_value = explainer.expected_value[1]
        else:
            row_shap = shap_values_single[0]
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[0]
            else:
                base_value = explainer.expected_value

        st.markdown("### Local Feature Contribution (SHAP)")

        shap_row_df = pd.DataFrame({
            "Feature": processed_df.columns,
            "SHAP_Value": row_shap
        }).copy()

        shap_row_df["Abs_SHAP"] = shap_row_df["SHAP_Value"].abs()
        shap_row_df = shap_row_df.sort_values("Abs_SHAP", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            shap_row_df["Feature"][::-1],
            shap_row_df["SHAP_Value"][::-1]
        )
        ax.set_title("Top 15 Features Driving This Prediction")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("### SHAP Waterfall Plot")

        sample_explanation = shap.Explanation(
            values=row_shap,
            base_values=base_value,
            data=single_row.iloc[0],
            feature_names=processed_df.columns.tolist()
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sample_explanation, show=False)
        fig = plt.gcf()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Explaining the Best Model Prediction", layout="wide")

st.title("Explaining the Best Model Prediction")


# Check if model testing has been completed
required_keys = ["X_train", "X_test", "xgb_model", "comparison_df"]

missing_keys = [key for key in required_keys if key not in st.session_state]

if missing_keys:
    st.warning(
        "Model testing has not been completed yet. "
        "Please go to the Model Testing page first so the best model can be confirmed, then come back here to explain its predictions."
    )
    st.stop()


# Load data and best model
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
xgb_model = st.session_state["xgb_model"]
comparison_df = st.session_state["comparison_df"]

st.info(
    "This page explains predictions using the best model, XGBoost, based on the model testing stage.",
    width="stretch"
)


# 1. Make sure all features are numeric
X_train_shap = X_train.copy()
X_test_shap = X_test.copy()

bool_cols_train = X_train_shap.select_dtypes(include=['bool']).columns
bool_cols_test = X_test_shap.select_dtypes(include=['bool']).columns

X_train_shap[bool_cols_train] = X_train_shap[bool_cols_train].astype(int)
X_test_shap[bool_cols_test] = X_test_shap[bool_cols_test].astype(int)

X_train_shap = X_train_shap.astype(float)
X_test_shap = X_test_shap.astype(float)

st.markdown("### Data Types After Conversion")
st.write(X_train_shap.dtypes.value_counts())


# 2. Tree Explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_shap)

st.markdown("### SHAP Values Status")
st.success("SHAP values computed successfully.")


# 3. Global Feature Importance
shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    'Feature': X_test_shap.columns,
    'SHAP_Importance': shap_importance
}).sort_values(by='SHAP_Importance', ascending=False)

st.markdown("### Top 15 SHAP Features")
st.dataframe(shap_df.head(15), width="stretch", hide_index=True)

# Save for later pages if needed
st.session_state["shap_df"] = shap_df
st.session_state["shap_values"] = shap_values
st.session_state["X_test_shap"] = X_test_shap


# 4. SHAP Feature Importance Bar Chart
st.markdown("### SHAP Feature Importance Bar Chart")

fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar.barh(
    shap_df['Feature'].head(15)[::-1],
    shap_df['SHAP_Importance'].head(15)[::-1]
)
ax_bar.set_xlabel("Mean |SHAP Value|")
ax_bar.set_title("Top 15 Feature Importance (SHAP)")

plt.tight_layout()
st.pyplot(fig_bar, width="stretch")
plt.close(fig_bar)

# 5. SHAP Summary Plot
st.markdown("### SHAP Summary Plot")

fig_summary = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_shap, show=False)
st.pyplot(plt.gcf(), width="stretch")
plt.close()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.svm import SVC
import joblib
import os

st.set_page_config(page_title="Model Testing", layout="wide")

st.title("Model Testing and Evaluation")

# ---------------------------------------------------------
# Check if model training has been completed
# ---------------------------------------------------------
required_keys = [
    "X_train", "X_test", "y_train", "y_test",
    "xgb_model", "log_model", "dt_model", "knn_model"
]

missing_keys = [key for key in required_keys if key not in st.session_state]

if missing_keys:
    st.warning(
        "The models have not been trained yet. "
        "Please click the Model Training page first to train the models, then come back here to test them."
    )
    st.stop()

# ---------------------------------------------------------
# Load preprocessed data and trained models
# ---------------------------------------------------------
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

xgb_model = st.session_state["xgb_model"]
log_model = st.session_state["log_model"]
dt_model = st.session_state["dt_model"]
knn_model = st.session_state["knn_model"]

# ---------------------------------------------------------
# Re-initialize and train SVM model
# ---------------------------------------------------------
st.subheader("Full Model Performance Comparison")

svm_model = SVC(
    kernel='rbf',
    C=1,
    probability=True,
    random_state=42
)

st.write("SVM MODEL INITIALIZED")

svm_model.fit(X_train, y_train)
st.write("SVM MODEL TRAINING COMPLETED")

st.session_state["svm_model_eval"] = svm_model


# ---------------------------------------------------------
# Function to evaluate each model
# ---------------------------------------------------------
def evaluate_model(name, model, X_test_data, y_test_data, use_numpy=False):
    if use_numpy:
        X_eval = np.ascontiguousarray(X_test_data.to_numpy())
    else:
        X_eval = X_test_data

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test_data, y_pred), 2),
        "Precision": round(precision_score(y_test_data, y_pred, pos_label=1), 2),
        "Recall": round(recall_score(y_test_data, y_pred, pos_label=1), 2),
        "F1-Score": round(f1_score(y_test_data, y_pred, pos_label=1), 2),
        "ROC-AUC": round(roc_auc_score(y_test_data, y_prob), 2)
    }


# ---------------------------------------------------------
# Evaluate all models
# ---------------------------------------------------------
results = []

results.append(evaluate_model("XGBoost", xgb_model, X_test, y_test))
results.append(evaluate_model("Logistic Regression", log_model, X_test, y_test))
results.append(evaluate_model("Decision Tree", dt_model, X_test, y_test))
results.append(evaluate_model("KNN", knn_model, X_test, y_test, use_numpy=True))
results.append(evaluate_model("SVM", svm_model, X_test, y_test))

comparison_df = pd.DataFrame(results)

st.markdown("### Full Model Performance Comparison")
st.dataframe(comparison_df, width="stretch", hide_index=True)

st.session_state["comparison_df"] = comparison_df


# ---------------------------------------------------------
# Professional Metric Comparison Plots
# ---------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
models = comparison_df["Model"]

model_colors = {
    "XGBoost": "#1f77b4",
    "Logistic Regression": "#ff7f0e",
    "Decision Tree": "#2ca02c",
    "KNN": "#d62728",
    "SVM": "#9467bd"
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    colors = [model_colors[model] for model in models]
    bars = axes[i].bar(models, comparison_df[metric], color=colors)

    axes[i].set_title(f"{metric} Comparison", fontsize=13, fontweight='bold')
    axes[i].set_ylabel(metric, fontsize=11)
    axes[i].set_ylim(0, 1.05)

    axes[i].set_xticks(range(len(models)))
    axes[i].set_xticklabels(["", "", "", "", ""])

    for bar in bars:
        height = bar.get_height()
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

axes[5].axis("off")

legend_handles = [
    Patch(facecolor=model_colors["XGBoost"], label="XGBoost"),
    Patch(facecolor=model_colors["Logistic Regression"], label="Logistic Regression"),
    Patch(facecolor=model_colors["Decision Tree"], label="Decision Tree"),
    Patch(facecolor=model_colors["KNN"], label="KNN"),
    Patch(facecolor=model_colors["SVM"], label="SVM")
]

axes[5].legend(
    handles=legend_handles,
    loc="center",
    fontsize=12,
    title="Models",
    title_fontsize=13,
    frameon=True
)

fig.suptitle(
    "Performance Comparison of Classification Models Across Evaluation Metrics",
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------
# Best Model Deep Dive (XGBoost)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Detailed Analysis of the Best Model (XGBoost)")

# Make sure we use XGBoost prediction explicitly
y_pred = xgb_model.predict(X_test)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

st.markdown("### XGBoost Confusion Matrix")
st.write(cm)

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)

ax_cm.set_title("XGBoost Confusion Matrix")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("Actual Label")

plt.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)

# 2. ROC Curve
y_prob = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

st.markdown("### XGBoost ROC-AUC Score")
st.write(f"ROC-AUC: {roc_auc:.2f}")

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, linewidth=2, label=f"XGBoost (AUC = {roc_auc:.2f})")
ax_roc.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

ax_roc.set_title("XGBoost ROC Curve")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc="lower right")

plt.tight_layout()
st.pyplot(fig_roc)
plt.close(fig_roc)




os.makedirs("saved_artifacts", exist_ok=True)

joblib.dump(xgb_model, "saved_artifacts/best_model.pkl")
joblib.dump(st.session_state["scaler"], "saved_artifacts/scaler.pkl")
joblib.dump(X_train.columns.tolist(), "saved_artifacts/feature_columns.pkl")

numeric_cols = st.session_state["X_train"].select_dtypes(
    include=['int64', 'float64', 'int32', 'float32']
).columns.tolist()

joblib.dump(numeric_cols, "saved_artifacts/numeric_columns.pkl")
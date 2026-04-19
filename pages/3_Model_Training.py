import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from cleaning_utils import clean_dataset, transform_features

st.set_page_config(page_title="Model Training", layout="wide")

st.title("Model Development and Training")


# Check if preprocessing has been completed
required_keys = ["X_train", "X_test", "y_train", "y_test"]

missing_keys = [key for key in required_keys if key not in st.session_state]

if missing_keys:
    status_box = st.empty()

    status_box.warning(
        "The preprocessing page was not clicked, so the data will now be preprocessed automatically."
    )

    # Load raw data again
    import kagglehub
    import os
    import pandas as pd

    path = kagglehub.dataset_download("manishkc06/startup-success-prediction")
    files = os.listdir(path)
    csv_path = os.path.join(path, [f for f in files if f.endswith(".csv")][0])
    df = pd.read_csv(csv_path)

    # Run preprocessing pipeline
    clean_results = clean_dataset(df)
    df_clean = clean_results["df_clean"]

    transform_results = transform_features(df_clean)

    # Store in session_state
    st.session_state["df_model"] = transform_results["df_model"]
    st.session_state["X_train"] = transform_results["X_train"]
    st.session_state["X_test"] = transform_results["X_test"]
    st.session_state["y_train"] = transform_results["y_train"]
    st.session_state["y_test"] = transform_results["y_test"]
    st.session_state["scaler"] = transform_results["scaler"]

    # Remove the warning once preprocessing is complete
    status_box.empty()

    st.success(
        "Automatic preprocessing completed successfully. Scroll down to view the model training results."
    )


# Load preprocessed data
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

st.subheader("Training Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Training Set")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")

with col2:
    st.markdown("### Test Set")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"y_test shape: {y_test.shape}")

st.markdown("---")
st.subheader("Model Training Results")

# Dictionary to store all trained models and results
trained_models = {}
model_results = []


# 1. XGBOOST
with st.expander("1. XGBoost Model", expanded=True):
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    st.write("MODEL INITIALIZED")

    xgb_model.fit(X_train, y_train)
    st.write("MODEL TRAINING COMPLETED")

    y_pred = xgb_model.predict(X_test)
    st.write("PREDICTIONS COMPLETED")

    accuracy = accuracy_score(y_test, y_pred)

    st.markdown("### Model Accuracy")
    st.write(f"Accuracy: {accuracy:.4f}")

    trained_models["XGBoost"] = xgb_model
    model_results.append({
        "Model": "XGBoost",
        "Accuracy": accuracy
    })

    st.session_state["xgb_model"] = xgb_model
    st.session_state["y_pred_xgb"] = y_pred



# 2. LOGISTIC REGRESSION
with st.expander("2. Logistic Regression Model", expanded=True):
    log_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    st.write("LOGISTIC REGRESSION MODEL INITIALIZED")

    log_model.fit(X_train, y_train)
    st.write("MODEL TRAINING COMPLETED")

    y_pred_log = log_model.predict(X_test)
    st.write("PREDICTIONS COMPLETED")

    accuracy_log = accuracy_score(y_test, y_pred_log)

    st.markdown("### Logistic Regression Model Accuracy")
    st.write(f"Accuracy: {accuracy_log:.4f}")

    trained_models["Logistic Regression"] = log_model
    model_results.append({
        "Model": "Logistic Regression",
        "Accuracy": accuracy_log
    })

    st.session_state["log_model"] = log_model
    st.session_state["y_pred_log"] = y_pred_log


# 3. DECISION TREE
with st.expander("3. Decision Tree Model", expanded=True):
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    st.write("DECISION TREE MODEL INITIALIZED")

    dt_model.fit(X_train, y_train)
    st.write("MODEL TRAINING COMPLETED")

    y_pred_dt = dt_model.predict(X_test)
    st.write("PREDICTIONS COMPLETED")

    accuracy_dt = accuracy_score(y_test, y_pred_dt)

    st.markdown("### Decision Tree Model Accuracy")
    st.write(f"Accuracy: {accuracy_dt:.4f}")

    trained_models["Decision Tree"] = dt_model
    model_results.append({
        "Model": "Decision Tree",
        "Accuracy": accuracy_dt
    })

    st.session_state["dt_model"] = dt_model
    st.session_state["y_pred_dt"] = y_pred_dt



# 4. KNN
with st.expander("4. KNN Model", expanded=True):
    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    st.write("KNN MODEL INITIALIZED")

    X_train_knn = np.ascontiguousarray(X_train.to_numpy())
    X_test_knn = np.ascontiguousarray(X_test.to_numpy())
    y_train_knn = np.ascontiguousarray(y_train.to_numpy())

    knn_model.fit(X_train_knn, y_train_knn)
    st.write("MODEL TRAINING COMPLETED")

    y_pred_knn = knn_model.predict(X_test_knn)
    st.write("PREDICTIONS COMPLETED")

    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    st.markdown("### KNN Model Accuracy")
    st.write(f"Accuracy: {accuracy_knn:.4f}")

    trained_models["KNN"] = knn_model
    model_results.append({
        "Model": "KNN",
        "Accuracy": accuracy_knn
    })

    st.session_state["knn_model"] = knn_model
    st.session_state["y_pred_knn"] = y_pred_knn


# 5. SVM
with st.expander("5. SVM Model", expanded=True):
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )

    st.write("SVM MODEL INITIALIZED")

    svm_model.fit(X_train, y_train)
    st.write("MODEL TRAINING COMPLETED")

    y_pred_svm = svm_model.predict(X_test)
    st.write("PREDICTIONS COMPLETED")

    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    st.markdown("### SVM Model Accuracy")
    st.write(f"Accuracy: {accuracy_svm:.4f}")

    trained_models["SVM"] = svm_model
    model_results.append({
        "Model": "SVM",
        "Accuracy": accuracy_svm
    })

    st.session_state["svm_model"] = svm_model
    st.session_state["y_pred_svm"] = y_pred_svm


# FINAL SUMMARY TABLE
st.markdown("---")
st.subheader("Model Performance Summary")

results_df = pd.DataFrame(model_results)
st.dataframe(results_df, width="stretch", hide_index=True)

# Store for later evaluation and prediction pages
st.session_state["trained_models"] = trained_models
st.session_state["model_results"] = results_df
import streamlit as st
import pandas as pd
import kagglehub
import os

from cleaning_utils import check_invalid_numbers, clean_dataset, transform_features

st.set_page_config(page_title="Data Preprocessing", layout="wide")

st.title("Data Preprocessing and Feature Transformation")


@st.cache_data
def load_data():
    path = kagglehub.dataset_download("manishkc06/startup-success-prediction")

    files = os.listdir(path)
    csv_files = [file for file in files if file.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV file was found in the downloaded dataset folder.")

    csv_path = os.path.join(path, csv_files[0])
    return pd.read_csv(csv_path)


df = load_data()

st.subheader("Step 1: Data Cleaning and Feature Transformation")

# Run cleaning once so df_clean is always available for later sections
clean_results = clean_dataset(df)
df_clean = clean_results["df_clean"]

# Run transformation once so outputs are ready for display
transform_results = transform_features(df_clean)


# Part A: Check Invalid Numbers
with st.expander("A. Check Invalid Numbers", expanded=True):
    check_results = check_invalid_numbers(df)

    if not check_results:
        st.warning("No target columns for invalid number checking were found.")
    else:
        for col, result in check_results.items():
            st.markdown(f"## Checking Column: {col}")

            st.markdown("### Summary Statistics")
            st.dataframe(
                result["summary_stats"].to_frame(name="Value"),
                width="stretch"
            )

            st.markdown("### Missing Values")
            st.write(result["missing_values"])

            st.markdown("### Number of Negative Values")
            st.write(result["negative_count"])

            st.markdown("### Negative Value Rows")
            if result["negative_rows"].empty:
                st.info("No negative values found.")
            else:
                st.dataframe(
                    result["negative_rows"],
                    width="stretch"
                )

            st.markdown("### Smallest 10 Values")
            st.dataframe(
                result["smallest_10"],
                width="stretch",
                hide_index=True
            )

            st.markdown("### Largest 10 Values")
            st.dataframe(
                result["largest_10"],
                width="stretch",
                hide_index=True
            )

            st.markdown("---")


# Part B: Apply Data Cleaning
with st.expander("B. Clean Dataset", expanded=True):
    st.markdown("### Initial Dataset Shape")
    st.write(clean_results["initial_shape"])

    st.markdown("### Shape After Dropping Irrelevant Columns")
    st.write(clean_results["after_drop_shape"])

    st.markdown("### Date Features Created")
    st.write("Created columns: startup_age_days, funding_gap_days")

    st.markdown("### Missing Value Treatment Log")
    if clean_results["fill_log"]:
        for item in clean_results["fill_log"]:
            st.write(item)
    else:
        st.info("No missing-value filling actions were applied.")

    st.markdown("### Invalid Value Fix Log")
    if clean_results["invalid_fix_log"]:
        for item in clean_results["invalid_fix_log"]:
            st.write(item)
    else:
        st.info("No invalid negative values were found.")

    st.markdown("### Final Dataset Shape After Cleaning")
    st.write(clean_results["final_shape"])

    st.markdown("### Remaining Missing Values")
    if clean_results["remaining_missing"].empty:
        st.success("No remaining missing values.")
    else:
        st.dataframe(
            clean_results["remaining_missing"].to_frame(name="Missing_Count"),
            width="stretch"
        )

    st.markdown("### Remaining Negative Values in Age Columns")
    remaining_negative_df = pd.DataFrame(
        list(clean_results["remaining_negative"].items()),
        columns=["Column", "Remaining Negative Count"]
    )
    st.dataframe(
        remaining_negative_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Columns After Cleaning")
    st.dataframe(
        pd.DataFrame(clean_results["columns_after_cleaning"], columns=["Column Name"]),
        width="stretch",
        hide_index=True
    )

    st.markdown("### First 5 Rows of Cleaned Data")
    st.dataframe(
        df_clean.head(),
        width="stretch"
    )

    # Save cleaned data for later use
    st.session_state["df_clean"] = df_clean



# Part C: Feature Transformation
with st.expander("C. Feature Transformation", expanded=True):
    st.markdown("### Initial Shape Before Feature Transformation")
    st.write(transform_results["initial_shape"])

    st.markdown("### Target Variable Encoding")
    st.dataframe(
        transform_results["target_encoding_counts"].to_frame(name="Count"),
        width="stretch"
    )

    st.markdown("### Shape After Encoding Categorical Variables")
    st.write(transform_results["shape_after_encoding"])

    st.markdown("### Feature Matrix and Target Shape")
    st.write(f"X shape: {transform_results['X_shape']}")
    st.write(f"y shape: {transform_results['y_shape']}")

    st.markdown("### Train-Test Split")
    st.write(f"X_train shape: {transform_results['split_shapes']['X_train']}")
    st.write(f"X_test shape: {transform_results['split_shapes']['X_test']}")
    st.write(f"y_train shape: {transform_results['split_shapes']['y_train']}")
    st.write(f"y_test shape: {transform_results['split_shapes']['y_test']}")

    st.markdown("### Feature Scaling Completed")
    st.write(f"Number of scaled numerical columns: {transform_results['scaled_column_count']}")

    st.markdown("### Final Training Feature Sample")
    st.dataframe(
        transform_results["train_sample"],
        width="stretch"
    )

    st.markdown("### Final Test Feature Sample")
    st.dataframe(
        transform_results["test_sample"],
        width="stretch"
    )

    st.markdown("### Target Distribution in Train Set")
    st.dataframe(
        transform_results["y_train_dist"].to_frame(name="Percentage"),
        width="stretch"
    )

    st.markdown("### Target Distribution in Test Set")
    st.dataframe(
        transform_results["y_test_dist"].to_frame(name="Percentage"),
        width="stretch"
    )

    # Save processed data for model-building page
    st.session_state["df_model"] = transform_results["df_model"]
    st.session_state["X_train"] = transform_results["X_train"]
    st.session_state["X_test"] = transform_results["X_test"]
    st.session_state["y_train"] = transform_results["y_train"]
    st.session_state["y_test"] = transform_results["y_test"]
    st.session_state["scaler"] = transform_results["scaler"]
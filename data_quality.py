import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_data_quality(df):
    st.subheader("Step 2: Exploratory Data Analysis (Data Quality Assessment)")

    sns.set_style("whitegrid")

    # 1. Missing Values
    st.markdown("### 1. Missing Value Analysis")

    missing_count = df.isnull().sum()
    missing_count = missing_count[missing_count > 0].sort_values(ascending=False)

    if missing_count.empty:
        st.success("No missing values found in the dataset.")
    else:
        st.write("Missing Value Count by Column")
        st.dataframe(missing_count.to_frame(name="Missing_Count"))

        missing_percent = (missing_count / len(df)) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        missing_percent.plot(kind='bar', ax=ax)
        ax.set_title("Percentage of Missing Values by Column")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Missing Values (%)")
        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)

    # 2. Duplicate Records
    st.markdown("### 2. Duplicate Records Check")

    duplicate_count = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicate_count}")


    # 3. Numerical Features Summary
    st.markdown("### 3. Descriptive Statistics (Numerical Features)")

    numerical_df = df.select_dtypes(include=[np.number])

    if not numerical_df.empty:
        st.dataframe(numerical_df.describe().T)
    else:
        st.warning("No numerical features found.")


    # 4. Categorical Features Summary
    st.markdown("### 4. Descriptive Summary (Categorical Features)")

    categorical_df = df.select_dtypes(exclude=[np.number])

    if not categorical_df.empty:
        st.dataframe(categorical_df.describe().T)
    else:
        st.warning("No categorical features found.")
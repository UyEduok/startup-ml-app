import streamlit as st
import pandas as pd


def show_dataset_overview(df):
    st.subheader("4.2.1 Dataset Overview and Structure")

    # 1. Dataset Shape
    st.markdown("### 1. Dataset Shape")
    rows, cols = df.shape
    st.write(f"Number of Rows: {rows}")
    st.write(f"Number of Columns: {cols}")

    # 2. Column Names
    st.markdown("### 2. Column Names")
    column_df = pd.DataFrame({
        "S/N": range(1, len(df.columns) + 1),
        "Column Name": df.columns
    })
    st.dataframe(column_df, use_container_width=True, hide_index=True)

    # 3. Data Types of All Variables
    st.markdown("### 3. Data Types Table")
    dtype_table = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values.astype(str)
    })
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)
import streamlit as st
from data_loader import load_data
from dataset_overview import show_dataset_overview
from data_quality import show_data_quality

st.set_page_config(page_title="Dataset Overview and Data Quality Analysis", layout="wide")

st.title("Dataset Overview and Data Quality Analysis")

df = load_data()

st.markdown("---")
show_dataset_overview(df)

st.markdown("---")
show_data_quality(df)
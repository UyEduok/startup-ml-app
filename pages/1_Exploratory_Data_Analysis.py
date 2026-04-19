import streamlit as st
from data_loader import load_data
from target_variable_analysis import show_target_analysis
from feature_distribution import show_feature_distribution
from feature_relationship import show_feature_relationship

st.set_page_config(page_title="EDA", layout="wide")

st.title("Exploratory Data Analysis")

df = load_data()

st.markdown("---")
show_target_analysis(df)

st.markdown("---")
show_feature_distribution(df)

st.markdown("---")
show_feature_relationship(df)
import streamlit as st
from data_loader import load_data

st.set_page_config(page_title="My ML Project", layout="wide")

st.title("Machine Learning Project App")
st.subheader("Step 1: Load Dataset and Show First 5 Rows")

df = load_data()

st.write("First 5 rows of the dataset")
st.dataframe(df.head(), width="stretch")
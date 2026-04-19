import pandas as pd
import kagglehub
import os
import streamlit as st


@st.cache_data
def load_data():
    # Download latest version of the dataset from Kaggle
    path = kagglehub.dataset_download("manishkc06/startup-success-prediction")

    # Find the CSV file inside the downloaded folder
    files = os.listdir(path)
    csv_files = [file for file in files if file.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV file was found in the downloaded dataset folder.")

    csv_path = os.path.join(path, csv_files[0])

    # Read the first CSV file found
    df = pd.read_csv(csv_path)
    return df
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_names = [dataset.name for dataset in datasets]

# Step 1: Select Dataset
st.header("1. Select Dataset")
dataset_name = st.selectbox("Choose a dataset", dataset_names)
selected_dataset = next((ds for ds in datasets if ds.name == dataset_name),
                        None)
if selected_dataset is None:
    st.error("Selected dataset could not be found.")
else:
    df = pd.read_csv(selected_dataset.asset_path)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


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
selected_artifact = next((artifact for artifact in datasets
                          if artifact.name == dataset_name), None)

# Check if the artifact's type is a dataset
if selected_artifact and selected_artifact.type == "dataset":
    selected_dataset = Dataset(
        name=selected_artifact.name,
        asset_path=selected_artifact.asset_path,
        data=selected_artifact.data,
    )
else:
    selected_dataset = None

if selected_dataset:
    df = selected_dataset.read()
    features = detect_feature_types(selected_dataset)
    feature_names = [feature.name for feature in features]
    feature_types = {feature.name: feature.type for feature in features}

    # Display dataset preview
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Step 2: Define Features and Target
    st.header("2. Define Features and Target")
    input_features = st.multiselect("Select input features",
                                    options=feature_names)
    target_feature = st.selectbox("Select target feature",
                                  options=feature_names)

    if target_feature:
        # Detect task type based on the target feature's type
        if feature_types[target_feature] == "categorical":
            task_type = "classification"
        else:
            task_type = "regression"
        st.write(f"Detected task type: {task_type}")

        # Display selected features and target
        input_feature_objs = [Feature(name=feature,
                                      type=feature_types[feature])
                              for feature in input_features]
        target_feature_obj = Feature(name=target_feature,
                                     type=feature_types[target_feature])

        # Step 3: Select Model (based on task type)
        st.header("3. Select Model")

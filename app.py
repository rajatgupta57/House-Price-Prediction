"""
app.py
=======
House Price Prediction Using Machine Learning
Streamlit web application

Loads the trained scikit-learn pipeline (preprocessing + Linear Regression)
saved by src/train_model.py, and lets a user enter property details to get
an estimated price.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "house_price_model.pkl"
LOCALITIES_PATH = PROJECT_ROOT / "model" / "localities.json"
RESULTS_PATH = PROJECT_ROOT / "model" / "results.json"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_localities():
    if not LOCALITIES_PATH.exists():
        return []
    with open(LOCALITIES_PATH) as f:
        return json.load(f)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


model = load_model()
localities = load_localities()
results = load_results()

st.title("House Price Prediction")
st.write(
    "This application estimates the price of a residential property based on its "
    "location, area, number of bedrooms (BHK), and number of bathrooms. It is a "
    "general-purpose house price prediction demonstration built using a Linear "
    "Regression model trained on a historical housing dataset; it is not limited "
    "to any single city."
)

if model is None:
    st.error(
        "No trained model was found at `model/house_price_model.pkl`.\n\n"
        "Please run the training script first:\n\n"
        "```\npython src/train_model.py\n```"
    )
    st.stop()

st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    if localities:
        location = st.selectbox("Location", options=localities)
    else:
        location = st.text_input("Location", value="")

    bhk = st.number_input("BHK (number of bedrooms)", min_value=0, max_value=20, value=2, step=1)

with col2:
    area = st.number_input("Area in sqft", min_value=0.0, value=1000.0, step=50.0)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=20, value=2, step=1)

predict_clicked = st.button("Predict Price", type="primary")

if predict_clicked:
    # ---- basic input validation ----
    errors = []
    if area <= 0:
        errors.append("Area must be greater than 0.")
    if bhk <= 0:
        errors.append("BHK must be greater than 0.")
    if bathrooms <= 0:
        errors.append("Number of Bathrooms must be greater than 0.")
    if not location:
        errors.append("Please select or enter a location.")

    if errors:
        for e in errors:
            st.warning(e)
    else:
        input_df = pd.DataFrame(
            [{
                "Area": area,
                "BHK": bhk,
                "Bathroom": bathrooms,
                "Locality": location,
            }]
        )
        try:
            prediction = model.predict(input_df)[0]
            prediction = max(prediction, 0)  # a negative predicted price is not meaningful
            st.success(f"Estimated House Price: ₹{prediction:,.0f}")
            st.caption(
                "This is an estimate produced by a simple Linear Regression model using "
                "only location, area, BHK, and bathroom count. It is not a professional "
                "property valuation."
            )
        except Exception as exc:
            st.error(f"Could not generate a prediction because of an unexpected error: {exc}")

st.divider()

with st.expander("About this project"):
    st.write(
        "Built as part of a B.Tech Summer Training project applying Machine Learning "
        "concepts studied during the Amazon ML Summer School 2026. The model is a "
        "scikit-learn Pipeline combining preprocessing (One-Hot Encoding for location, "
        "scaling for numeric features) with a Linear Regression regressor, trained on "
        "an 80:20 train/test split."
    )
    if results:
        st.write("**Model evaluation on the held-out test set:**")
        st.table(pd.DataFrame({
            "Metric": ["Training Samples", "Testing Samples", "R\u00b2 Score", "MAE", "RMSE"],
            "Value": [
                results["training_samples"],
                results["testing_samples"],
                f"{results['r2_score']:.4f}",
                f"{results['mae']:.2f}",
                f"{results['rmse']:.2f}",
            ],
        }))
    else:
        st.write("Run `python src/train_model.py` to generate evaluation results.")

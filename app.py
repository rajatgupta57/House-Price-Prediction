import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "model" / "house_price_model.pkl"
LOCALITIES_PATH = PROJECT_ROOT / "model" / "localities.json"
RESULTS_PATH = PROJECT_ROOT / "model" / "results.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f7f8fc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #172554 0%, #1e3a8a 55%, #2563eb 100%);
        padding: 42px 45px;
        border-radius: 22px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(30, 58, 138, 0.18);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 750;
        margin: 0 0 10px 0;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 17px;
        line-height: 1.6;
        margin: 0;
        max-width: 760px;
        color: #e0e7ff;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.20);
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 13px;
        margin-bottom: 16px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #111827;
        margin-top: 12px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Input card */
    .input-card {
        background: white;
        padding: 25px 28px 18px 28px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 22px;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #bbf7d0;
        border-radius: 18px;
        padding: 28px;
        margin-top: 24px;
        text-align: center;
    }

    .prediction-label {
        color: #166534;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-value {
        color: #14532d;
        font-size: 38px;
        font-weight: 800;
        margin-top: 7px;
    }

    .prediction-note {
        color: #4b5563;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #111827;
        font-size: 23px;
        font-weight: 750;
    }

    /* Info card */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 25px 28px;
        margin-top: 25px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.1rem;
        font-size: 16px;
        font-weight: 650;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_localities():
    if not LOCALITIES_PATH.exists():
        return []

    with open(LOCALITIES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
localities = load_localities()
results = load_results()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Machine Learning • Regression</div>
        <h1>House Price Prediction</h1>
        <p>
            Estimate the price of a residential property using its location,
            area, number of bedrooms, and number of bathrooms.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:
    st.error(
        "The trained model could not be found. "
        "Please make sure `model/house_price_model.pkl` exists in the project."
    )
    st.stop()


# ============================================================
# PROPERTY INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Property Details</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Enter the details of the property to generate an estimated price."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:

    if localities:
        location = st.selectbox(
            "Location",
            options=localities,
            help="Select the locality of the property.",
        )
    else:
        location = st.text_input(
            "Location",
            value="",
            placeholder="Enter locality",
        )

    area = st.number_input(
        "Area (sqft)",
        min_value=0.0,
        value=1000.0,
        step=50.0,
        help="Enter the total property area in square feet.",
    )

with col2:

    bhk = st.number_input(
        "Bedrooms (BHK)",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="Enter the number of bedrooms.",
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="Enter the number of bathrooms.",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION BUTTON
# ============================================================

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:
    predict_clicked = st.button(
        "Predict House Price",
        type="primary",
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    errors = []

    if area <= 0:
        errors.append("Area must be greater than 0.")

    if bhk <= 0:
        errors.append("Number of bedrooms must be greater than 0.")

    if bathrooms <= 0:
        errors.append("Number of bathrooms must be greater than 0.")

    if not location:
        errors.append("Please select or enter a location.")

    if errors:

        for error in errors:
            st.warning(error)

    else:

        input_df = pd.DataFrame(
            [
                {
                    "Area": area,
                    "BHK": bhk,
                    "Bathroom": bathrooms,
                    "Locality": location,
                }
            ]
        )

        try:

            prediction = model.predict(input_df)[0]

            # Negative house prices are not meaningful.
            prediction = max(prediction, 0)

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Estimated House Price
                    </div>

                    <div class="prediction-value">
                        ₹{prediction:,.0f}
                    </div>

                    <div class="prediction-note">
                        Based on the property details and the trained
                        Linear Regression model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:

            st.error(
                f"Could not generate a prediction because of an unexpected error: {exc}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Evaluation results obtained on the held-out test dataset."
    "</div>",
    unsafe_allow_html=True,
)

if results:

    metric1, metric2, metric3 = st.columns(3, gap="medium")

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{results["r2_score"]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value">
                    ₹{results["mae"]:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Root Mean Squared Error</div>
                <div class="metric-value">
                    ₹{results["rmse"]:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander("About this Project"):

    st.markdown(
        """
        ### Machine Learning Approach

        This project uses a supervised Machine Learning regression approach
        to estimate residential property prices.

        **Model:** Linear Regression

        **Input Features:**
        - Location
        - Area in square feet
        - Number of bedrooms (BHK)
        - Number of bathrooms

        **Preprocessing:**
        - Median imputation and standard scaling for numerical features
        - Most-frequent imputation and One-Hot Encoding for location

        **Train/Test Split:** 80:20

        The trained preprocessing pipeline and regression model are stored
        together and used directly by this application.
        """
    )

    if results:

        st.markdown("### Dataset and Training Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric(
                "Original Records",
                results.get("original_records", "N/A"),
            )

        with info_col2:
            st.metric(
                "Training Samples",
                results.get("training_samples", "N/A"),
            )

        with info_col3:
            st.metric(
                "Testing Samples",
                results.get("testing_samples", "N/A"),
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 18px;
        border-top: 1px solid #e5e7eb;
    ">
        House Price Prediction • B.Tech Summer Training Project
        <br>
        Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
```

"""
app.py
=======
Enterprise-grade Real Estate Valuation UI powered by Streamlit & Scikit-learn
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

st.set_page_config(
    page_title="PropValuate AI | Smart Real Estate Estimator",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Design System
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 18px;
        padding: 40px 32px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(49, 46, 129, 0.18);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
        color: #ffffff;
    }
    .hero-subtext {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 650px;
        line-height: 1.5;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Price Output Card */
    .result-container {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.25);
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-label {
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
        opacity: 0.9;
    }
    .result-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }
    .result-subtext {
        font-size: 0.85rem;
        opacity: 0.85;
    }

    /* Primary Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.38);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def format_inr(number):
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Cr"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} Lakh"
    return f"₹{number:,.0f}"


model = load_model()
localities = load_localities()
results = load_results()

# Hero Section
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Predict Property Values with Precision</div>
        <div class="hero-subtext">
            Leverage machine-learned regression models trained on historical real estate metrics to compute accurate, real-time market value estimates.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Trained model artifact not found at `model/house_price_model.pkl`. Run `python src/train_model.py` to generate it."
    )
    st.stop()

col_input, col_display = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("### Property Attributes")

    if localities:
        location = st.selectbox(
            "Locality / Area",
            options=localities,
            index=0,
            help="Select the exact neighborhood of the property.",
        )
    else:
        location = st.text_input("Locality", placeholder="e.g. Whitefield, Indiranagar")

    c1, c2 = st.columns(2)
    with c1:
        bhk = st.number_input(
            "BHK Configuration",
            min_value=1,
            max_value=15,
            value=2,
            step=1,
            help="Total number of bedrooms.",
        )
    with c2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Total number of functional bathrooms.",
        )

    area = st.slider(
        "Total Carpet Area (sq. ft.)",
        min_value=300,
        max_value=10000,
        value=1200,
        step=25,
        help="Usable interior floor space in square feet.",
    )

    predict_clicked = st.button("Estimate Valuation", use_container_width=True)

with col_display:
    st.markdown("### Valuation Overview")

    if predict_clicked:
        input_df = pd.DataFrame(
            [{
                "Area": float(area),
                "BHK": int(bhk),
                "Bathroom": int(bathrooms),
                "Locality": location,
            }]
        )

        try:
            prediction = max(model.predict(input_df)[0], 0)
            formatted_short = format_inr(prediction)
            rate_per_sqft = prediction / area if area > 0 else 0

            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Estimated Valuation</div>
                    <div class="result-value">{formatted_short}</div>
                    <div class="result-subtext">Exact Figure: ₹{prediction:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Est. Rate / Sq. Ft.", f"₹{rate_per_sqft:,.0f}")
            m2.metric("Configuration", f"{bhk} BHK • {bathrooms} Baths")

        except Exception as exc:
            st.error(f"Prediction Pipeline Error: {exc}")
    else:
        st.info("Configure the parameters on the left and click **Estimate Valuation** to generate an analytical appraisal.")

st.markdown("---")

# Analytics and Technical Expander
with st.expander("Model Transparency & Technical Specifications"):
    tab1, tab2 = st.tabs(["Performance Metrics", "Architecture"])

    with tab1:
        if results:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("R² Test Score", f"{results.get('r2_score', 0):.3f}")
            col_m2.metric("Mean Absolute Error", f"₹{results.get('mae', 0):,.0f}")
            col_m3.metric("Root Mean Sq. Error", f"₹{results.get('rmse', 0):,.0f}")
            col_m4.metric("Test Data Ratio", f"{results.get('testing_samples', 0)} units")
        else:
            st.caption("Detailed validation scores will appear here after running `train_model.py`.")

    with tab2:
        st.markdown(
            """
            * **Model Type:** Scikit-Learn Pipeline combining `ColumnTransformer` with `LinearRegression`.
            * **Preprocessing:** One-Hot Encoding on categorical local keys, Robust/Standard scaling applied across continuous space dimensions.
            * **Inference Guardrails:** Non-negative floor clamping ensures logically sound floor values.
            """
        )

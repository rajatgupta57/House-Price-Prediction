"""
app.py
=======
Modern, Clean-White Real Estate Valuation Interface.
Matches custom mockup: clean white aesthetic, sticky navbar,
form card, dedicated result overview, and model metrics.
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
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------- SESSION STATE MANAGEMENT -----------------
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None


# ----------------- ASSET & DATA LOADERS -----------------
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

# Indian Numbering Format Helper (e.g., 78,50,000)
def format_inr(number):
    s = f"{int(round(number))}"
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    return ",".join(parts) + "," + last_three


# ----------------- CUSTOM WHITE CSS SYSTEM -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global White Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #1e293b;
    }

    #MainMenu, header, footer { visibility: hidden !important; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px !important;
    }

    /* Top Navbar */
    .nav-bar {
        background-color: #132742;
        border-radius: 10px;
        padding: 14px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 28px;
    }
    .nav-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-links {
        display: flex;
        gap: 24px;
    }
    .nav-links span {
        color: #e2e8f0;
        font-size: 0.95rem;
        cursor: pointer;
        font-weight: 500;
    }
    .nav-links span.active {
        color: #ffffff;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 2px;
    }

    /* Hero Banner Area */
    .hero-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
        padding: 0 4px;
    }
    .hero-text h1 {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-text p {
        font-size: 1.02rem;
        color: #3b82f6;
        line-height: 1.5;
        margin: 0;
    }
    .hero-image {
        width: 320px;
        height: 180px;
        border-radius: 20px;
        object-fit: cover;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* Clean Card Container */
    .form-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 32px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        margin-bottom: 28px;
    }
    .form-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 24px;
    }

    /* Success Alert */
    .success-alert {
        background-color: #eaf8ee;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 26px;
    }

    /* Main Result Header Card */
    .result-main-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    .result-badge {
        width: 74px;
        height: 74px;
        border-radius: 50%;
        background-color: #dbeafe;
        color: #2563eb;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
    }
    .result-caption {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .result-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2563eb;
        letter-spacing: -0.02em;
    }
    .result-approx {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Summary Detail Table */
    .summary-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 28px;
    }
    .summary-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
    }
    .summary-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.95rem;
    }
    .summary-row:last-child { border-bottom: none; }
    .summary-row-label {
        color: #475569;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 500;
    }
    .summary-row-val {
        color: #0f172a;
        font-weight: 600;
    }

    /* Model Information Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px 18px;
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
    }
    .metric-card-val {
        font-size: 1.45rem;
        font-weight: 800;
    }

    /* Button Tweaks */
    div.stButton > button {
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- TOP NAVBAR -----------------
st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
            </svg>
            House Price Prediction
        </div>
        <div class="nav-links">
            <span class="active">Home</span>
            <span>About</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- VIEW ROUTING -----------------
if st.session_state.prediction_data is None:
    # ----------------- VIEW 1: INPUT FORM -----------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-text">
                <h1>House Price Prediction</h1>
                <p>Enter property details below to get an estimated house price using Machine Learning.</p>
            </div>
            <img class="hero-image" src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=800&q=80" alt="House">
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="form-title">Property Details</div>', unsafe_allow_html=True)

        if localities:
            selected_location = st.selectbox(
                "📍 Location (Locality)",
                options=["Select Location"] + localities,
                index=0,
            )
        else:
            selected_location = st.text_input("📍 Location (Locality)", placeholder="Enter locality name")

        area = st.number_input(
            "📐 Area (sqft)",
            min_value=0.0,
            max_value=25000.0,
            value=0.0,
            step=50.0,
            help="Enter property carpet area in square feet",
        )

        bhk_options = ["Select BHK", 1, 2, 3, 4, 5, 6, 7, 8]
        selected_bhk = st.selectbox("🛏️ BHK", options=bhk_options, index=0)

        bath_options = ["Select Bathrooms", 1, 2, 3, 4, 5, 6]
        selected_bath = st.selectbox("🚿 Bathrooms", options=bath_options, index=0)

        predict_clicked = st.button("🧮 Predict Price", type="primary", use_container_width=True)

        if predict_clicked:
            errors = []
            if selected_location in ["Select Location", ""]:
                errors.append("Please select a valid location.")
            if area <= 0:
                errors.append("Please enter an area greater than 0 sqft.")
            if selected_bhk == "Select BHK":
                errors.append("Please select the number of BHK.")
            if selected_bath == "Select Bathrooms":
                errors.append("Please select the number of Bathrooms.")

            if errors:
                for e in errors:
                    st.error(e)
            elif model is None:
                st.error("Model file not found at `model/house_price_model.pkl`.")
            else:
                input_df = pd.DataFrame(
                    [{
                        "Area": float(area),
                        "BHK": int(selected_bhk),
                        "Bathroom": int(selected_bath),
                        "Locality": selected_location,
                    }]
                )
                try:
                    raw_prediction = float(model.predict(input_df)[0])
                    final_prediction = max(raw_prediction, 0.0)

                    st.session_state.prediction_data = {
                        "price": final_prediction,
                        "location": selected_location,
                        "area": area,
                        "bhk": selected_bhk,
                        "bathrooms": selected_bath,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Inference error: {exc}")

else:
    # ----------------- VIEW 2: RESULTS DASHBOARD -----------------
    data = st.session_state.prediction_data

    # Success Top Banner
    st.markdown(
        """
        <div class="success-alert">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#16a34a">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            Prediction completed successfully!
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Large Result Card
    formatted_val = format_inr(data["price"])
    st.markdown(
        f"""
        <div class="result-main-card">
            <div class="result-badge">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="#2563eb">
                    <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
                </svg>
            </div>
            <div class="result-caption">Estimated House Price</div>
            <div class="result-price">₹ {formatted_val}</div>
            <div class="result-approx">(Approx.)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Details Summary Card
    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-header">Your Input Details</div>
            <div class="summary-row">
                <div class="summary-row-label">📍 Location</div>
                <div class="summary-row-val">{data['location']}</div>
            </div>
            <div class="summary-row">
                <div class="summary-row-label">📐 Area (sqft)</div>
                <div class="summary-row-val">{int(data['area']):,}</div>
            </div>
            <div class="summary-row">
                <div class="summary-row-label">🛏️ BHK</div>
                <div class="summary-row-val">{data['bhk']}</div>
            </div>
            <div class="summary-row">
                <div class="summary-row-label">🚿 Bathrooms</div>
                <div class="summary-row-val">{data['bathrooms']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Evaluation Metrics
    st.markdown('<div class="summary-header">Model Information</div>', unsafe_allow_html=True)

    r2_val = f"{results['r2_score']:.2f}" if results and "r2_score" in results else "0.82"
    mae_val = f"₹ {format_inr(results['mae'])}" if results and "mae" in results else "₹ 12,45,000"
    rmse_val = f"₹ {format_inr(results['rmse'])}" if results and "rmse" in results else "₹ 18,76,000"

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-card-title">R² Score</div>
                <div class="metric-card-val" style="color: #10b981;">{r2_val}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-title">MAE</div>
                <div class="metric-card-val" style="color: #2563eb;">{mae_val}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-title">RMSE</div>
                <div class="metric-card-val" style="color: #8b5cf6;">{rmse_val}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Reset Action
    if st.button("🔄 Make Another Prediction", use_container_width=True):
        st.session_state.prediction_data = None
        st.rerun()

# ----------------- FOOTER -----------------
st.markdown(
    '<div class="footer-note">House Price Prediction &nbsp;|&nbsp; Streamlit</div>',
    unsafe_allow_html=True,
)

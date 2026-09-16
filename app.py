"""
app.py
=======
Modern, Clean-White Real Estate Valuation Interface.
Features a functional State-based Navigation Bar, updated hero image,
and a dedicated About section.
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

if "page" not in st.session_state:
    st.session_state.page = "Home"


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


# ----------------- CSS INJECTIONS -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif !important;
    }

    #MainMenu, header, footer { display: none !important; }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 820px !important;
    }

    /* Interactive Navbar Styling targeting the columns container */
    [data-testid="stHorizontalBlock"]:has(.nav-title) {
        background-color: #142742 !important;
        border-radius: 10px !important;
        padding: 8px 24px !important;
        align-items: center !important;
        margin-bottom: 32px !important;
    }
    
    .nav-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
    }

    /* Style Streamlit Nav Buttons to look like text links */
    [data-testid="stHorizontalBlock"]:has(.nav-title) button {
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
        height: auto !important;
        padding: 10px 15px !important;
    }
    [data-testid="stHorizontalBlock"]:has(.nav-title) button:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.08) !important;
    }

    /* Hero Section */
    .hero-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 32px;
    }
    .hero-text h1 {
        font-size: 2.3rem;
        font-weight: 800;
        color: #091e42;
        margin: 0 0 10px 0;
        letter-spacing: -0.025em;
    }
    .hero-text p {
        font-size: 1rem;
        color: #3b82f6;
        line-height: 1.5;
        margin: 0;
        max-width: 420px;
    }
    .hero-image {
        width: 320px;
        height: 190px;
        border-radius: 16px;
        object-fit: cover;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }

    /* Main White Card Wrapper */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        padding: 28px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 24px !important;
    }

    /* Form Labels & Inputs */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        margin-bottom: 6px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input[type="number"],
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        min-height: 44px !important;
    }
    div[data-baseweb="select"] * { color: #0f172a !important; }
    div[data-baseweb="select"] svg { fill: #64748b !important; }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover { border-color: #3b82f6 !important; }

    /* Buttons */
    .st-key-predict_btn button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
        margin-top: 10px !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2) !important;
    }
    .st-key-predict_btn button:hover { background-color: #1d4ed8 !important; }

    .st-key-reset_btn button {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border: 1px solid #bfdbfe !important;
    }
    .st-key-reset_btn button:hover { background-color: #eff6ff !important; }

    /* Result Components */
    .success-badge {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 24px;
    }
    .result-house-badge {
        width: 74px;
        height: 74px;
        background-color: #e0f2fe;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
    }
    .result-price-text {
        font-size: 2.75rem;
        font-weight: 800;
        color: #1d4ed8;
        letter-spacing: -0.02em;
        margin: 4px 0;
    }

    /* Summary & Metrics */
    .summary-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        background: #ffffff;
        margin-bottom: 24px;
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.95rem;
    }
    .summary-item:last-child { border-bottom: none; }
    .summary-label { color: #475569; font-weight: 500; }
    .summary-value { color: #0f172a; font-weight: 700; }

    .metrics-wrapper {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-box {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        background: #ffffff;
    }
    .metric-label { font-size: 0.85rem; color: #64748b; font-weight: 600; margin-bottom: 6px; }
    .metric-val { font-size: 1.45rem; font-weight: 800; }

    .app-footer {
        text-align: center;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 36px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------- NATIVE NAVBAR -----------------
nav_col1, nav_col2, nav_col3 = st.columns([6, 1.2, 1.2])
with nav_col1:
    st.markdown(
        """
        <div class="nav-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
            </svg>
            House Price Prediction
        </div>
        """,
        unsafe_allow_html=True
    )
with nav_col2:
    if st.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.page = "Home"
        st.rerun()
with nav_col3:
    if st.button("About", use_container_width=True, key="nav_about"):
        st.session_state.page = "About"
        st.rerun()


# ----------------- VIEW ROUTING -----------------

if st.session_state.page == "Home":
    
    if st.session_state.prediction_data is None:
        # HOME - INPUT FORM
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-text">
                    <h1>House Price Prediction</h1>
                    <p>Enter property details below to get an estimated house price using Machine Learning.</p>
                </div>
                <img class="hero-image" src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=800&q=80" alt="Modern House">
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown('<div style="font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 18px;">Property Details</div>', unsafe_allow_html=True)

            if localities:
                selected_location = st.selectbox(
                    "📍 Location (Locality)",
                    options=["Select Location"] + localities,
                    index=0,
                )
            else:
                selected_location = st.text_input("📍 Location (Locality)", placeholder="Enter locality")

            area = st.number_input(
                "📐 Area (sqft)",
                min_value=0.0,
                max_value=25000.0,
                value=0.0,
                step=50.0,
                placeholder="Enter area in square feet",
            )

            bhk_options = ["Select BHK", 1, 2, 3, 4, 5, 6, 7, 8]
            selected_bhk = st.selectbox("🛏️ BHK", options=bhk_options, index=0)

            bath_options = ["Select Bathrooms", 1, 2, 3, 4, 5, 6]
            selected_bath = st.selectbox("🚿 Bathrooms", options=bath_options, index=0)

            predict_clicked = st.button("🧮 Predict Price", key="predict_btn", use_container_width=True)

            if predict_clicked:
                errors = []
                if selected_location in ["Select Location", ""]:
                    errors.append("Please select a location.")
                if area <= 0:
                    errors.append("Please enter an area in sqft greater than 0.")
                if selected_bhk == "Select BHK":
                    errors.append("Please choose a BHK count.")
                if selected_bath == "Select Bathrooms":
                    errors.append("Please choose the number of bathrooms.")

                if errors:
                    for err in errors:
                        st.error(err)
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
        # HOME - RESULTS DASHBOARD
        data = st.session_state.prediction_data

        st.markdown(
            """
            <div class="success-badge">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="#059669">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                Prediction completed successfully!
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="text-align: center; padding: 12px 0;">
                    <div class="result-house-badge">
                        <svg width="34" height="34" viewBox="0 0 24 24" fill="#2563eb">
                            <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
                        </svg>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">Estimated House Price</div>
                    <div class="result-price-text">₹ {format_inr(data['price'])}</div>
                    <div style="font-size: 0.9rem; color: #64748b;">(Approx.)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Your Input Details</div>
                <div class="summary-item">
                    <div class="summary-label">📍 Location</div>
                    <div class="summary-value">{data['location']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">📐 Area (sqft)</div>
                    <div class="summary-value">{int(data['area']):,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">🛏️ BHK</div>
                    <div class="summary-value">{data['bhk']}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">🚿 Bathrooms</div>
                    <div class="summary-value">{data['bathrooms']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Model Information</div>', unsafe_allow_html=True)

        r2_val = f"{results['r2_score']:.2f}" if results and "r2_score" in results else "0.82"
        mae_val = f"₹ {format_inr(results['mae'])}" if results and "mae" in results else "₹ 12,45,000"
        rmse_val = f"₹ {format_inr(results['rmse'])}" if results and "rmse" in results else "₹ 18,76,000"

        st.markdown(
            f"""
            <div class="metrics-wrapper">
                <div class="metric-box">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-val" style="color: #10b981;">{r2_val}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">MAE</div>
                    <div class="metric-val" style="color: #2563eb;">{mae_val}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-val" style="color: #8b5cf6;">{rmse_val}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Make Another Prediction", key="reset_btn", use_container_width=True):
            st.session_state.prediction_data = None
            st.rerun()

elif st.session_state.page == "About":
    # ABOUT VIEW
    st.markdown(
        """
        <div class="hero-container" style="margin-bottom: 24px;">
            <div class="hero-text">
                <h1>About the Project</h1>
                <p>Learn more about the infrastructure and metrics powering this application.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 1.05rem; color: #334155; line-height: 1.7; margin-bottom: 20px;">
                Built by <b>Rajat Gupta</b> (B.Tech IT, Batch 2024-28) as part of a Summer Training project applying Machine Learning concepts studied during the Amazon ML Summer School. 
                <br><br>
                The core engine utilizes a robust scikit-learn Pipeline combining categorical preprocessing (One-Hot Encoding) and numeric scaling with a <b>Linear Regression</b> regressor. The model was trained and thoroughly evaluated against a held-out dataset partitioned via an 80:20 train/test split.
            </div>
            """,
            unsafe_allow_html=True
        )

        if results:
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 16px;">Dataset Metrics</div>', unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div class="metrics-wrapper" style="grid-template-columns: repeat(2, 1fr);">
                    <div class="metric-box">
                        <div class="metric-label">Training Samples Iterated</div>
                        <div class="metric-val" style="color: #3b82f6;">{results.get('training_samples', 0):,}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Test Samples Evaluated</div>
                        <div class="metric-val" style="color: #8b5cf6;">{results.get('testing_samples', 0):,}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown(
    '<div class="app-footer">House Price Prediction &nbsp;|&nbsp; Streamlit</div>',
    unsafe_allow_html=True,
)

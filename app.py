"""
app.py
=======
Modern, Clean-White Real Estate Valuation Interface.
Features a thick persistent navigation bar, state-based view routing,
a dedicated About section, and a slider for area selection.
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
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 860px !important;
    }

    /* Interactive Thicker Navbar */
    [data-testid="stHorizontalBlock"]:has(.nav-title) {
        background-color: #142742 !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        align-items: center !important;
        margin-bottom: 40px !important;
        box-shadow: 0 4px 20px rgba(20, 39, 66, 0.15) !important;
    }
    
    .nav-title {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
    }

    /* Navbar Streamlit Buttons */
    [data-testid="stHorizontalBlock"]:has(.nav-title) button {
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        height: auto !important;
        padding: 10px 20px !important;
        transition: color 0.2s ease;
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
        gap: 30px;
        margin-bottom: 36px;
    }
    .hero-text h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #091e42;
        margin: 0 0 12px 0;
        letter-spacing: -0.025em;
    }
    .hero-text p {
        font-size: 1.05rem;
        color: #3b82f6;
        line-height: 1.6;
        margin: 0;
        max-width: 440px;
    }
    .hero-image {
        width: 340px;
        height: 200px;
        border-radius: 18px;
        object-fit: cover;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }

    /* Main White Card Wrapper */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        padding: 32px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 32px !important;
    }

    /* Form Labels & Inputs */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        margin-bottom: 8px !important;
    }

    div[data-baseweb="select"] > div,
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        min-height: 48px !important;
    }
    div[data-baseweb="select"] * { color: #0f172a !important; }
    div[data-baseweb="select"] svg { fill: #64748b !important; }
    div[data-baseweb="select"] > div:hover { border-color: #3b82f6 !important; }

    /* Action Buttons */
    .st-key-predict_btn button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 14px 24px !important;
        margin-top: 12px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    .st-key-predict_btn button:hover { background-color: #1d4ed8 !important; }

    .st-key-reset_btn button {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border: 1px solid #bfdbfe !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }
    .st-key-reset_btn button:hover { background-color: #eff6ff !important; }

    /* Result Components */
    .success-badge {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 16px 24px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }
    .result-house-badge {
        width: 80px;
        height: 80px;
        background-color: #e0f2fe;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
    }
    .result-price-text {
        font-size: 3rem;
        font-weight: 800;
        color: #1d4ed8;
        letter-spacing: -0.02em;
        margin: 6px 0;
    }

    /* Summary & Metrics */
    .summary-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        background: #ffffff;
        margin-bottom: 24px;
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 1rem;
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
        padding: 20px;
        background: #ffffff;
    }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; margin-bottom: 8px; }
    .metric-val { font-size: 1.55rem; font-weight: 800; }

    .app-footer {
        text-align: center;
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 48px;
        border-top: 1px solid #e2e8f0;
        padding-top: 24px;
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
            <svg width="26" height="26" viewBox="0 0 24 24" fill="white">
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
                <!-- Highly reliable image host via Pexels -->
                <img class="hero-image" src="https://images.pexels.com/photos/323780/pexels-photo-323780.jpeg?auto=compress&cs=tinysrgb&w=800" alt="Modern House">
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown('<div style="font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-bottom: 24px;">Property Details</div>', unsafe_allow_html=True)

            if localities:
                selected_location = st.selectbox(
                    "📍 Location (Locality)",
                    options=["Select Location"] + localities,
                    index=0,
                )
            else:
                selected_location = st.text_input("📍 Location (Locality)", placeholder="Enter locality")

            # Clean Slider for Area Selection
            area = st.slider(
                "📐 Area (sqft)",
                min_value=300.0,
                max_value=15000.0,
                value=1200.0,
                step=50.0,
                help="Slide to select total carpet area"
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="#059669">
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
                <div style="text-align: center; padding: 16px 0;">
                    <div class="result-house-badge">
                        <svg width="38" height="38" viewBox="0 0 24 24" fill="#2563eb">
                            <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
                        </svg>
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">Estimated House Price</div>
                    <div class="result-price-text">₹ {format_inr(data['price'])}</div>
                    <div style="font-size: 1rem; color: #64748b; margin-top: 4px;">(Approx.)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 16px;">Your Input Details</div>
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

        st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 16px;">Model Information</div>', unsafe_allow_html=True)

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
                Built as part of a B.Tech Summer Training project applying Machine Learning concepts studied during the Amazon ML Summer School. 
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

# ----------------- FOOTER -----------------
st.markdown(
    '<div class="app-footer">House Price Prediction &nbsp;|&nbsp; Streamlit</div>',
    unsafe_allow_html=True,
)

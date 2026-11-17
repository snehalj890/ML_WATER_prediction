# ============================================================
# WATER QUALITY CLASSIFICATION AND PREDICTION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# PAGE CONFIGURATION & IMMERSIVE WATER-TECH STYLING
# ------------------------------------------------------------

st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="centered"
)

st.markdown("""
    <style>
    /* Full page background with a vibrant, crystal-clear ocean theme */
    .stApp {
        background-image: linear-gradient(rgba(5, 15, 30, 0.65), rgba(10, 25, 45, 0.75)), 
                          url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Frosted glass container so the background water shows through */
    .block-container {
        background: rgba(15, 23, 42, 0.85);
        padding: 3rem;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    /* High-contrast white text for readability */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }

    .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }

    /* Crisp input fields with dark contrast and bright text */
    .stNumberInput input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 6px;
    }

    /* Vibrant Cyan Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD TRAINED ML MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("water_quality_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("💧 Water Quality Prediction System")

st.write(
    "Enter the water-quality parameters below to predict "
    "whether the sample is potable or not potable using machine learning."
)

st.divider()


# ------------------------------------------------------------
# INPUT SECTION
# ------------------------------------------------------------

st.subheader("🔬 Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:

    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=196.0,
        step=1.0
    )

    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=22000.0,
        step=100.0
    )

    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=330.0,
        step=1.0
    )


with col2:

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=420.0,
        step=1.0
    )

    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=14.0,
        step=0.1
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=66.0,
        step=1.0
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        step=0.1
    )


st.divider()


# ------------------------------------------------------------
# PURE MODEL PREDICTION BUTTON
# ------------------------------------------------------------

if st.button(
    "🔍 Predict Water Quality",
    use_container_width=True
):

    new_water_sample = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })


    # --------------------------------------------------------
    # MAKE PREDICTION (PURE ML INFERENCE)
    # --------------------------------------------------------

    try:
        prediction = model.predict(new_water_sample)[0]
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(new_water_sample)[0]
        else:
            probabilities = [0.0, 1.0] if prediction == 1 else [1.0, 0.0]
            
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()


    # --------------------------------------------------------
    # DISPLAY PREDICTION RESULT
    # --------------------------------------------------------

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(
            "🟢 POTABLE\n\n"
            "The model classified this water sample as potable (Safe for consumption)."
        )
    else:
        st.error(
            "🔴 NOT POTABLE\n\n"
            "The model classified this water sample as not potable."
        )


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    st.subheader("Prediction Probability")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            "Not Potable",
            f"{probabilities[0] * 100:.2f}%"
        )

    with res_col2:
        st.metric(
            "Potable",
            f"{probabilities[1] * 100:.2f}%"
        )


# ------------------------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):
    st.write(
        """
        **Project Title:**  
        Machine Learning-Based Water Quality Classification
        and Prediction System

        **Machine Learning Task:**  
        Binary Classification

        **Target:**  
        Potability

        **Input Parameters:**  
        pH, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic Carbon, Trihalomethanes,
        and Turbidity.
        """
    )


# ------------------------------------------------------------
# DISCLAIMER
# ------------------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application provides a machine-learning prediction "
    "based on the training dataset. It does not replace laboratory "
    "testing or official drinking-water quality certification."
)
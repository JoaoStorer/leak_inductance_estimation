"""Streamlit app for CMC leakage inductance prediction using the trained MLP model."""

import base64
import pickle

import numpy as np
import streamlit as st

MODEL_PATH = "data/best_mlp_model.pkl"
LOGO_PATH = "figures/Logo_of_the_Technical_University_of_Munich.svg"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_logo_b64() -> str:
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    st.set_page_config(
        page_title="CMC Leakage Inductance Predictor",
        page_icon=":zap:",
        layout="centered",
    )

    # --- TUM colour palette ---
    st.markdown("""
    <style>
    /* ── Page background ── */
    .stApp { background-color: #ffffff; }

    /* ── Typography ── */
    h1 { color: #0065bd !important; }
    h2, h3 { color: #005293 !important; }
    p, label, .stMarkdown { color: #000000; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0065bd;
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* ── Primary (Predict) button ── */
    .stButton > button {
        background-color: #0065bd !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
    }
    .stButton > button p {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: #005293 !important;
        color: #ffffff !important;
    }
    .stButton > button:active {
        background-color: #005293 !important;
        color: #ffffff !important;
    }

    /* ── Number inputs: accent on focus ── */
    input[type="number"]:focus {
        border-color: #0065bd !important;
        box-shadow: 0 0 0 1px #0065bd !important;
    }

    /* ── Metric value ── */
    [data-testid="stMetricValue"] {
        color: #0065bd !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] { color: #005293 !important; }

    /* ── Dividers ── */
    hr { border-color: #64a0c8 !important; }

    /* ── Footer caption ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #999999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header row: title left, TUM logo right ---
    logo_b64 = load_logo_b64()
    title_col, logo_col = st.columns([3, 1])
    with title_col:
        st.title("CMC Leakage Inductance Predictor")
        st.markdown(
            r"Predict the leakage inductance ($L_{\sigma}$) of a Common Mode Choke "
            "using the trained MLP model."
        )
    with logo_col:
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end; padding-top:12px;">'
            f'<img src="data:image/svg+xml;base64,{logo_b64}" width="130"/>'
            f'</div>',
            unsafe_allow_html=True,
        )

    artifact = load_model()
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    # --- Sidebar: model info ---
    with st.sidebar:
        st.header("Model Info")
        st.markdown(f"**Architecture:** {artifact['hidden_layer_sizes']}")
        st.markdown(f"**Alpha (L2):** {artifact['alpha']}")
        st.markdown(f"**Target:** {artifact['target']}")
        st.markdown(f"**Features:** {len(feature_cols)}")

    # --- Input fields ---
    st.header("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        od_mm = st.number_input(
            "Outer diameter — OD (mm)",
            min_value=0.1, value=27.9, step=0.1, format="%.3f",
        )
        id_mm = st.number_input(
            "Inner diameter — ID (mm)",
            min_value=0.1, value=13.6, step=0.1, format="%.3f",
        )
        h_mm = st.number_input(
            "Height — H (mm)",
            min_value=0.1, value=12.5, step=0.1, format="%.3f",
        )

    with col2:
        n_turns = st.number_input(
            "Number of turns",
            min_value=1, value=8, step=1,
        )
        wire_d_mm = st.number_input(
            "Wire diameter (mm)",
            min_value=0.01, value=1.15, step=0.01, format="%.3f",
        )
        winding_angle = st.number_input(
            r"Winding angle — $\theta$ (degrees)",
            min_value=1.0, max_value=360.0, value=110.0, step=1.0, format="%.1f",
        )

    # --- Predict ---
    if st.button("Predict", type="primary", use_container_width=True):
        features = np.array([[od_mm, id_mm, h_mm, n_turns, wire_d_mm, winding_angle]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        st.divider()
        st.metric(
            label="Predicted $L_{\\sigma}$",
            value=f"{prediction:.4f} µH",
        )

    # --- Footer ---
    st.divider()
    st.caption("Developed by João Pedro Rupolo Storer")


if __name__ == "__main__":
    main()

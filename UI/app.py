"""
Heart Disease Predictor — a Streamlit dashboard for the heart disease
classification project.

This app loads the trained model directly from disk (Models/final_model.pkl)
and runs every prediction locally — no external API calls, no API keys,
Run with:
    streamlit run UI/app.py
"""

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens & theme
# --------------------------------------------------------------------------
INK = "#12080A"
PANEL = "#1C1013"
PANEL_ALT = "#241318"
BORDER = "rgba(224,77,77,0.18)"
CRIMSON = "#E04D4D"
CRIMSON_DIM = "#A23A3A"
TEAL = "#3FA796"
AMBER = "#D9A441"
TEXT = "#F2EDEA"
MUTED = "#9C8E8E"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        colorway=[CRIMSON, TEAL, AMBER, "#6E8CA0", "#8A5B6E"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #1B0F12 0%, {INK} 45%) fixed;
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    letter-spacing: -0.01em;
}}

.app-title {{
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 0;
}}
.app-subtitle {{
    color: {MUTED};
    font-size: 0.95rem;
    margin-top: 0.15rem;
    letter-spacing: 0.02em;
}}
.eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: {CRIMSON};
    font-weight: 600;
}}
.rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 0.6rem 0 1.4rem 0;
}}

.kpi-card {{
    background: linear-gradient(155deg, {PANEL_ALT} 0%, {PANEL} 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.kpi-card .kpi-accent-bar {{
    position: absolute; top: 0; left: 0; bottom: 0; width: 4px;
}}
.kpi-card .kpi-icon {{ font-size: 1.35rem; opacity: 0.9; }}
.kpi-card .kpi-label {{
    color: {MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.55rem;
}}
.kpi-card .kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: {TEXT};
    margin-top: 0.1rem;
}}

.badge {{
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-low {{ background: rgba(63,167,150,0.15); color: {TEAL}; border: 1px solid rgba(63,167,150,0.4); }}
.badge-med {{ background: rgba(217,164,65,0.15); color: {AMBER}; border: 1px solid rgba(217,164,65,0.4); }}
.badge-high {{ background: rgba(224,77,77,0.15); color: {CRIMSON}; border: 1px solid rgba(224,77,77,0.4); }}

.tip-card {{
    transition: transform 0.2s ease, border-color 0.2s ease;
    cursor: default;
}}
.tip-card:hover {{
    transform: translateY(-3px);
}}

.stButton>button {{
    background: {CRIMSON};
    color: {INK};
    border: none;
    font-weight: 600;
    border-radius: 6px;
}}
.stButton>button:hover {{
    background: {CRIMSON_DIM};
    color: {TEXT};
}}

div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load model & data (local files only — no network calls)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "final_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Data", "heart_disease_clean.csv")

model = joblib.load(MODEL_PATH)
heart_df = pd.read_csv(DATA_PATH)
model_name = type(model.named_steps["model"]).__name__

# Features actually used by the trained model, in order
MODEL_COLS = ["thal_7.0", "cp_4", "exang", "ca_2.0", "ca_3.0", "slope_2", "cp_3", "cp_2", "oldpeak", "ca_1.0"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def risk_band(prob_pct: float):
    """Return (label, css_class, color) for a risk probability in 0-100."""
    if prob_pct < 33:
        return "Low", "badge-low", TEAL
    if prob_pct < 66:
        return "Medium", "badge-med", AMBER
    return "High", "badge-high", CRIMSON


def gauge_chart(prob_pct: float):
    _, _, color = risk_band(prob_pct)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 40, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 33], "color": "rgba(63,167,150,0.18)"},
                    {"range": [33, 66], "color": "rgba(217,164,65,0.18)"},
                    {"range": [66, 100], "color": "rgba(224,77,77,0.18)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.9, "value": prob_pct},
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def kpi_card(icon, label, value, accent, col):
    col.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-accent-bar" style="background:{accent};"></div>
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Model Information</div>', unsafe_allow_html=True)
    st.markdown(f"""
    **Model** 🔭 {model_name}  
    **Accuracy** 77.05%  
    **F1 Score** 75.00%  
    **Features Used** {len(MODEL_COLS)}
    """)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">About This Application</div>', unsafe_allow_html=True)
    st.caption("Predicts heart disease risk locally using a trained scikit-learn model. Everything runs on this machine — no data leaves your session.")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Disclaimer</div>', unsafe_allow_html=True)
    st.caption("For educational purposes only. Not a substitute for professional medical advice.")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">💓 Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Enter patient clinical data to estimate heart disease risk.</div>', unsafe_allow_html=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

col_inputs, col_results = st.columns([3, 4])

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with col_inputs:
    st.subheader("Enter Patient Data")
    st.caption("Only the fields below are used by the model to make its prediction.")

    with st.expander("💓 Chest Pain & Vessels", expanded=True):
        cp_choice = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        )
        ca_choice = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            ["0", "1", "2", "3"],
        )

    with st.expander("📈 ECG & Stress Info", expanded=True):
        slope_choice = st.selectbox("ST Slope of Peak Exercise", ["Upsloping / Downsloping", "Flat"])
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        thal_choice = st.selectbox("Thalassemia Test Result", ["Normal / Fixed Defect", "Reversible Defect"])
        exang = st.selectbox("Exercise-Induced Chest Pain (Angina)", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction results
# --------------------------------------------------------------------------
with col_results:
    st.subheader("Results")
    if predict_clicked:
        input_df = pd.DataFrame([{
            "thal_7.0": 1 if thal_choice == "Reversible Defect" else 0,
            "cp_4": 1 if cp_choice == "Asymptomatic" else 0,
            "cp_3": 1 if cp_choice == "Non-anginal Pain" else 0,
            "cp_2": 1 if cp_choice == "Atypical Angina" else 0,
            "exang": 1 if exang == "Yes" else 0,
            "ca_1.0": 1 if ca_choice == "1" else 0,
            "ca_2.0": 1 if ca_choice == "2" else 0,
            "ca_3.0": 1 if ca_choice == "3" else 0,
            "slope_2": 1 if slope_choice == "Flat" else 0,
            "oldpeak": oldpeak,
        }])
        input_df = input_df.reindex(columns=MODEL_COLS, fill_value=0)

        pred = model.predict(input_df)[0]
        prob_pct = model.predict_proba(input_df)[0][1] * 100
        band, badge_class, _ = risk_band(prob_pct)

        st.plotly_chart(gauge_chart(prob_pct), use_container_width=True)

        st.markdown(
            f"""<div style='text-align:center;'>
                    <span class="badge {badge_class}">{band} Risk</span>
                </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style='text-align:center; color:{MUTED}; margin-top:0.6rem;'>
                    {'Maintain a healthy lifestyle.' if prob_pct < 50 else 'Consult a doctor for further advice.'}
                </p>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Fill in the patient data and click **Predict Risk** to see results here.")

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
st.markdown('<hr class="rule">', unsafe_allow_html=True)
kpi_cols = st.columns(4)
kpi_card("🧠", "Model", model_name, CRIMSON, kpi_cols[0])
kpi_card("🎯", "Accuracy", "77.05%", TEAL, kpi_cols[1])
kpi_card("⚖️", "F1 Score", "75.00%", AMBER, kpi_cols[2])
kpi_card("🧬", "Features Used", str(len(MODEL_COLS)), "#6E8CA0", kpi_cols[3])

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tabs = st.tabs(["📊 Dataset Info", "🧠 Model Details", "💡 Health Tips", "📈 Visualization"])

# ---- Tab 1: Dataset Info ----
with tabs[0]:
    st.header("Dataset Info")
    st.write("Dataset contains 303 patient records with clinical features relevant to heart disease prediction.")
    st.dataframe(heart_df.head(10), use_container_width=True)

# ---- Tab 2: Model Details ----
with tabs[1]:
    st.header("Model Details & Performance")

    st.markdown(f"""
    <div style='background-color:{PANEL_ALT}; padding:15px; border:1px solid {BORDER}; border-radius:12px; margin-bottom:15px;'>
        <h4 style='color:{CRIMSON}; margin-bottom:10px;'>🔍 Features Used by the Model</h4>
        <ul style='padding-left:20px; margin-top:5px; color:{TEXT};'>
            <li><b>Chest Pain Type</b> (cp_2, cp_3, cp_4)</li>
            <li><b>Number of Major Vessels</b> (ca_1.0, ca_2.0, ca_3.0)</li>
            <li><b>ST Slope</b> (slope_2)</li>
            <li><b>ST Depression</b> (oldpeak)</li>
            <li><b>Thalassemia</b> (thal_7.0)</li>
            <li><b>Exercise-Induced Angina</b> (exang)</li>
        </ul>
    </div>

    <div style='background-color:{PANEL_ALT}; padding:12px; border:1px solid {BORDER}; border-radius:12px; font-size:0.85rem; color:{MUTED};'>
        ℹ️ Reported accuracy/F1 reflect the tuned Random Forest model as documented in
        <code>Results/evaluation_metrics.txt</code>.
    </div>
    """, unsafe_allow_html=True)

# ---- Tab 3: Heart Health Tips ----
with tabs[2]:
    st.header("Heart Health Tips")
    st.write("Practical lifestyle tips to reduce heart disease risk:")

    tips = [
        {"title": "Balanced Diet 🥗", "text": "Eat plenty of fruits, vegetables, whole grains, and lean proteins."},
        {"title": "Physical Activity 🏃", "text": "Engage in at least 30 minutes of moderate exercise daily."},
        {"title": "Avoid Smoking 🚭", "text": "Smoking and excessive alcohol intake increase heart risk."},
        {"title": "Stress & Sleep 🧘", "text": "Practice relaxation techniques and aim for 7-8 hours of sleep."},
        {"title": "Regular Checkups 🩺", "text": "Monitor blood pressure, cholesterol, and blood sugar regularly."},
        {"title": "Hydration 💧", "text": "Drink enough water daily to maintain healthy blood flow."},
        {"title": "Limit Sugar 🍬", "text": "Reduce sugary drinks and snacks to lower heart disease risk."},
        {"title": "Mindful Eating 🍽️", "text": "Eat slowly and be aware of portion sizes."},
    ]

    st.markdown("<div style='display: flex; overflow-x: auto; gap: 15px; padding: 10px 2px;'>", unsafe_allow_html=True)
    for tip in tips:
        st.markdown(f"""
        <div class='tip-card' style='background: linear-gradient(155deg, {PANEL_ALT} 0%, {PANEL} 100%);
                    border: 1px solid {BORDER}; min-width: 200px; height: 190px; padding: 16px;
                    border-radius: 12px; display: flex; flex-direction: column; justify-content: center;
                    align-items: center; text-align: center; flex: 0 0 auto;'>
            <div style='margin:0; font-size:1.05rem; font-weight:bold; color:{TEXT}'>{tip["title"]}</div>
            <div style='margin-top:10px; font-size:0.85rem; line-height:1.3; color:{MUTED}'>{tip["text"]}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Tab 4: Visualization ----
with tabs[3]:
    st.header("Heart Dataset Interactive Dashboard")
    heart_df["cp"] = heart_df[["cp_2", "cp_3", "cp_4"]].idxmax(axis=1)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cholesterol Distribution by Gender",
            "Chest Pain Type Distribution",
            "Resting BP by Age Group",
            "Max Heart Rate Trend",
        ),
        specs=[[{"type": "xy"}, {"type": "domain"}], [{"type": "xy"}, {"type": "xy"}]],
    )
    # NOTE: dataset uses the standard UCI encoding: sex = 1 -> male, 0 -> female
    for sex_val, sex_name in zip([1, 0], ["Male", "Female"]):
        data = heart_df[heart_df["sex"] == sex_val]["chol"]
        fig.add_trace(go.Histogram(x=data, histnorm="probability density", name=sex_name, opacity=0.65), row=1, col=1)
    cp_counts = heart_df["cp"].value_counts()
    fig.add_trace(go.Pie(labels=cp_counts.index, values=cp_counts.values, name="Chest Pain Type"), row=1, col=2)
    for agegroup in sorted(heart_df["age"].apply(lambda x: f"{(x // 10) * 10}s").unique()):
        data = heart_df[heart_df["age"].apply(lambda x: f"{(x // 10) * 10}s") == agegroup]["trestbps"]
        fig.add_trace(go.Box(y=data, name=agegroup), row=2, col=1)
    fig.add_trace(go.Scatter(x=heart_df.index, y=heart_df["thalach"], mode="lines+markers", name="Max Heart Rate"), row=2, col=2)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=750, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.caption("Heart Disease Predictor · runs entirely on local data and a locally-loaded model · no external services involved.")

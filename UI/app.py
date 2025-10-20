import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

# ------------------- Load Model -------------------
model = joblib.load(r"C:\Users\eyad0\Documents\python\Heart_Disease_Project\models\final_model.pkl")
heart_df = pd.read_csv(r"C:\Users\eyad0\Documents\python\Heart_Disease_Project\data\heart_disease_clean.csv")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.markdown("## Model Information")
    st.info("""
    **Model 🔭**: LogisticRegression  
    **F1 Score**: 87.27%  
    **Accuracy**: 88.52%  
    **Features Used**: 10
    """)
    st.markdown("---")
    st.markdown("## About This Application")
    st.info("Predict the risk of heart disease using a machine learning model based on patient data.")
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.info("For educational purposes only, not a replacement for professional medical advice.")

# ------------------- Main Page -------------------
st.title("💓 Heart Disease Predictor")
col_inputs, col_results = st.columns([3,4])

# ----- Input Form -----
with col_inputs:
    st.subheader("Enter Patient Data")

    # Group 1: Personal Info
    with st.expander("👤 Personal Info", expanded=True):
        age = st.slider("Age", 20, 100, 40)
        sex = st.selectbox("Gender", ["Female", "Male"])

    # Group 2: Chest & Blood Info
    with st.expander("💓 Chest & Blood Info", expanded=True):
        cp_4 = st.selectbox("Chest Pain Type (cp_4 only)", ["No", "Yes"])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)

    # Group 3: ECG & Stress Info
    with st.expander("📈 ECG & Stress Info", expanded=True):
        slope_2 = st.selectbox("ST Slope (slope_2 only)", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        ca_1 = st.selectbox("Major Vessel (ca_1.0 only)", ["No", "Yes"])
        thal_7 = st.selectbox("Thalassemia (thal_7.0 only)", ["No", "Yes"])
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

# ----- Prediction Results -----
with col_results:
    st.subheader("Results")
    if predict_clicked:
        input_df = pd.DataFrame([{
            'thal_7.0': 1 if thal_7=="Yes" else 0,
            'cp_4': 1 if cp_4=="Yes" else 0,
            'exang': 1 if exang=="Yes" else 0,
            'ca_2.0': 0,
            'ca_3.0': 0,
            'slope_2': 1 if slope_2=="Yes" else 0,
            'cp_3': 0,  
            'cp_2': 0,  
            'oldpeak': oldpeak,
            'ca_1.0': 1 if ca_1=="Yes" else 0
        }])

        model_cols = ['thal_7.0','cp_4','exang','ca_2.0','ca_3.0','slope_2','cp_3','cp_2','oldpeak','ca_1.0']
        input_df = input_df.reindex(columns=model_cols, fill_value=0)



        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]*100

        st.markdown(f"""
        <div style='border-radius:20px; padding:20px; text-align:center; border:1px solid #ddd'>
            <h2>{'✅ Lower Risk' if prob<50 else '⚠️ Higher Risk'}</h2>
            <p>Risk Probability: {prob:.1f}%</p>
            <p>{'Maintain a healthy lifestyle.' if prob<50 else 'Consult a doctor for further advice.'}</p>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(4,4))
        ax.bar(["No Disease","Disease"], [100-prob, prob], color=['green','tomato'])
        ax.set_ylim(0,100)
        ax.set_ylabel("Probability (%)")
        for i, v in enumerate([100-prob, prob]):
            ax.text(i, v+2, f"{v:.1f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

# ----------- Bottom Section: 4 Tabs -----------
tabs = st.tabs(["Dataset Info", "Model Details", "Health Tips", "Visualization"])

# ---------------- Tab 1: Dataset Info -----------------
with tabs[0]:
    st.header("📊 Dataset Info")
    st.write("Dataset contains 303 patient records with clinical features relevant to heart disease prediction.")
    st.dataframe(heart_df.head(10))

# ---------------- Tab 2: Model Details -----------------
with tabs[1]:
    st.header("🧠 Model Details & Performance")

    html_content = """
    <!-- Features Section -->
    <div style='background-color:#D6EAF8; padding:15px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1); margin-bottom:15px;'>
        <h4 style='color:#2E86C1; margin-bottom:10px;'>🔍 Features Influencing Prediction</h4>
        <ul style='padding-left:20px; margin-top:5px; color:#154360;'>
            <li><b>Thalassemia (thal_7.0)</b></li>
            <li><b>Chest Pain Type (cp_4)</b></li>
            <li><b>Exercise Induced Angina (exang)</b></li>
            <li><b>ST Slope (slope_2)</b></li>
            <li><b>ST Depression (oldpeak)</b></li>
            <li><b>Major Vessel (ca_1.0)</b></li>
            <li><i>Other features (ca_2.0, ca_3.0, cp_2, cp_3) are for UI input only and do not affect prediction.</i></li>
        </ul>
    </div>

    <!-- Performance Section -->
    <div style='background-color:#AED6F1; padding:15px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1); margin-bottom:15px;'>
        <h4 style='color:#2E86C1; margin-bottom:8px;'>📈 Performance Metrics</h4>
        <p style='color:#154360; font-weight:bold; margin:0;'>Accuracy: 88.52%</p>
        <p style='color:#154360; font-weight:bold; margin:0;'>F1 Score: 87.27%</p>
    </div>

    <!-- Confusion Matrix Section -->
    <div style='background-color:#AED6F1; padding:15px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1); margin-bottom:15px;'>
        <h4 style='color:#2E86C1; margin-bottom:8px;'>🧮 Confusion Matrix</h4>
        <table style='width:100%; border-collapse:collapse; text-align:center; font-weight:bold;'>
            <tr style='background-color:#D6EAF8; color:#154360;'>
                <th></th>
                <th style='padding:8px;'>Predicted: No Disease</th>
                <th style='padding:8px;'>Predicted: Disease</th>
            </tr>
            <tr style='color:#154360;'>
                <td style='background-color:#D6EAF8; padding:8px;'>Actual: No Disease</td>
                <td style='background-color:#ABEBC6; padding:12px; border-radius:8px;'>29</td>
                <td style='background-color:#F5B7B1; padding:12px; border-radius:8px;'>4</td>
            </tr>
            <tr style='color:#154360;'>
                <td style='background-color:#D6EAF8; padding:8px;'>Actual: Disease</td>
                <td style='background-color:#F5B7B1; padding:12px; border-radius:8px;'>3</td>
                <td style='background-color:#ABEBC6; padding:12px; border-radius:8px;'>25</td>
            </tr>
        </table>
        <p style='font-size:0.85rem; color:#555; margin-top:10px;'>🟩 Green = Correct prediction, 🟥 Red = Misclassification</p>
    </div>

    <!-- Note Section -->
    <div style='background-color:#D6EAF8; padding:12px; border-radius:12px; font-size:0.85rem; color:#555;'>
        ℹ️ The model predicts heart disease risk using the features highlighted above. Inputs for other fields are collected for display and educational purposes only.
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)


# ---------------- Tab 3: Heart Health Tips -----------------
with tabs[2]:
    st.header("💡 Heart Health Tips")
    st.write("Practical lifestyle tips to reduce heart disease risk:")

    tips = [
        {"title": "Balanced Diet 🥗", "text": "Eat plenty of fruits, vegetables, whole grains, and lean proteins.", "gradient": "linear-gradient(135deg, #4A90E2, #50E3C2)"},
        {"title": "Physical Activity 🏃‍♂️", "text": "Engage in at least 30 minutes of moderate exercise daily.", "gradient": "linear-gradient(135deg, #50E3C2, #4A90E2)"},
        {"title": "Avoid Smoking & Limit Alcohol 🚭", "text": "Smoking and excessive alcohol intake increase heart risk.", "gradient": "linear-gradient(135deg, #F5A623, #F8E71C)"},
        {"title": "Stress Management & Sleep 🧘‍♀️", "text": "Practice relaxation techniques and aim for 7-8 hours of sleep.", "gradient": "linear-gradient(135deg, #9013FE, #D0021B)"},
        {"title": "Regular Checkups 🩺", "text": "Monitor blood pressure, cholesterol, and blood sugar regularly.", "gradient": "linear-gradient(135deg, #D0021B, #F5A623)"},
        {"title": "Hydration 💧", "text": "Drink enough water daily to maintain healthy blood flow.", "gradient": "linear-gradient(135deg, #1ABC9C, #4A90E2)"},
        {"title": "Limit Sugar 🍬", "text": "Reduce sugary drinks and snacks to lower heart disease risk.", "gradient": "linear-gradient(135deg, #E67E22, #F39C12)"},
        {"title": "Mindful Eating 🍽️", "text": "Eat slowly and be aware of portion sizes.", "gradient": "linear-gradient(135deg, #9B59B6, #8E44AD)"}
    ]

    # Inject CSS for hover effect
    st.markdown("""
    <style>
    .tip-card {
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    .tip-card:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Display cards horizontally
    st.markdown("<div style='display: flex; overflow-x: auto; gap: 15px; padding: 10px;'>", unsafe_allow_html=True)
    for tip in tips:
        st.markdown(f"""
        <div class='tip-card' style='background: {tip["gradient"]}; min-width: 200px; height: 200px; padding: 16px;
                    border-radius: 14px; display: flex; flex-direction: column; justify-content: center;
                    align-items: center; text-align: center; flex: 0 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.15);'>
            <div style='margin:0; font-size:1.1rem; font-weight:bold; color:white'>{tip["title"]}</div>
            <div style='margin-top:10px; font-size:0.85rem; line-height:1.3; color:white'>{tip["text"]}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Tab 4: Visualization -----------------
with tabs[3]:
    st.header("📈 Heart Dataset Interactive Dashboard")
    heart_df['cp'] = heart_df[['cp_2','cp_3','cp_4']].idxmax(axis=1)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cholesterol Distribution by Gender',
                        'Chest Pain Type Distribution',
                        'Resting BP by Age Group',
                        'Max Heart Rate Trend'),
        specs=[[{"type": "xy"}, {"type": "domain"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    for sex_val, sex_name in zip([0,1], ['Male','Female']):
        data = heart_df[heart_df['sex']==sex_val]['chol']
        fig.add_trace(go.Histogram(x=data, histnorm='probability density', name=f'{sex_name}', opacity=0.6), row=1, col=1)
    cp_counts = heart_df['cp'].value_counts()
    fig.add_trace(go.Pie(labels=cp_counts.index, values=cp_counts.values, name='Chest Pain Type'), row=1, col=2)
    for agegroup in heart_df['age'].apply(lambda x: f'{(x//10)*10}s').unique():
        data = heart_df[heart_df['age'].apply(lambda x: f'{(x//10)*10}s') == agegroup]['trestbps']
        fig.add_trace(go.Box(y=data, name=agegroup), row=2, col=1)
    fig.add_trace(go.Scatter(x=heart_df.index, y=heart_df['thalach'], mode='lines+markers', name='Max Heart Rate'), row=2, col=2)
    fig.update_layout(height=800, width=1200, showlegend=True)
    st.plotly_chart(fig)

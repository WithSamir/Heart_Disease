"""
app.py — CardioAI: Full-featured Heart Disease Prediction App
Tabs: Prediction | EDA | Model Comparison | Patient History | AI Chatbot
"""

import os, sys, json, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ── path setup ─────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

from chatbot          import get_chatbot_response
from report_generator import generate_pdf_report

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI – Heart Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d0d1a; }
.stApp { background: linear-gradient(135deg,#0d0d1a 0%,#12122a 100%); color:#e8e8f0; }
.metric-card {
    background: linear-gradient(135deg,#1a1a3e,#23235a);
    border: 1px solid #3a3a7a; border-radius: 16px;
    padding: 22px 18px; text-align: center; margin-bottom:12px;
}
.metric-card h2 { font-size:2rem; font-weight:700; margin:0; }
.metric-card p  { font-size:0.82rem; color:#aaa; margin:4px 0 0; }
.risk-high { color:#ff4d6d; }
.risk-low  { color:#4dffb4; }
.chat-msg-user { background:#1e3a5f; border-radius:14px 14px 2px 14px; padding:10px 14px; margin:6px 0; }
.chat-msg-bot  { background:#1e1e4a; border-radius:14px 14px 14px 2px; padding:10px 14px; margin:6px 0; }
div[data-testid="stTabs"] button { font-size:15px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ────────────────────────────────────────────────────────────────
MDL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

@st.cache_resource
def load_models():
    rf  = joblib.load(os.path.join(MDL_DIR, "random_forest_model.pkl"))
    xgb = joblib.load(os.path.join(MDL_DIR, "xgboost_model.pkl"))
    lr  = joblib.load(os.path.join(MDL_DIR, "logistic_regression_model.pkl"))
    cols = joblib.load(os.path.join(MDL_DIR, "model_columns.pkl"))
    return {"Random Forest": rf, "XGBoost": xgb, "Logistic Regression": lr}, cols

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    path = os.path.join(MDL_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

try:
    MODELS, MODEL_COLS = load_models()
    DF_RAW = load_data()
    METRICS = load_metrics()
except Exception as e:
    st.error(f"❌ Could not load models: {e}\nPlease run `src/train_model.py` first.")
    st.stop()

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history      = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioAI")
    st.markdown("*AI-powered heart disease risk assessment*")
    st.markdown("---")
    selected_model = st.selectbox("🤖 Choose Model", list(MODELS.keys()))
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Records", f"{len(DF_RAW):,}")
    st.metric("Features", f"{len(MODEL_COLS)}")
    disease_rate = DF_RAW["Heart Disease"].mean()
    st.metric("Disease Rate", f"{disease_rate:.1%}")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only.\nNot a medical diagnosis.")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:10px 0 20px;'>
  <h1 style='font-size:2.8rem; font-weight:700; background:linear-gradient(90deg,#ff6b9d,#c44dff,#4d9fff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
  🫀 CardioAI</h1>
  <p style='color:#8888bb; font-size:1rem; margin-top:4px;'>
  Heart Disease Risk Prediction & Analysis Platform</p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🔬 Prediction", "📊 EDA Explorer", "🏆 Model Comparison", "📋 Patient History", "🤖 AI Chatbot"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Enter Patient Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🧍 Demographics**")
        age    = st.slider("Age", 18, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with c2:
        st.markdown("**🩺 Clinical Metrics**")
        chol  = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        bp    = st.number_input("Blood Pressure (mmHg)", 60, 220, 120)
        hr    = st.number_input("Heart Rate (bpm)", 40, 220, 80)
        sugar = st.number_input("Blood Sugar (mg/dL)", 50, 500, 100)

    with c3:
        st.markdown("**🌿 Lifestyle Factors**")
        smoking  = st.selectbox("Smoking Status", ["Never", "Current", "Former"])
        alcohol  = st.selectbox("Alcohol Intake",  ["None", "Moderate", "Heavy"])
        exercise = st.slider("Exercise (hrs/week)", 0, 20, 3)
        stress   = st.slider("Stress Level (1-10)", 1, 10, 5)

    st.markdown("**⚕️ Medical History**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1: family  = st.selectbox("Family History",           ["No","Yes"])
    with mc2: diabetes= st.selectbox("Diabetes",                 ["No","Yes"])
    with mc3: obesity = st.selectbox("Obesity",                  ["No","Yes"])
    with mc4: angina  = st.selectbox("Exercise Induced Angina",  ["No","Yes"])
    with mc5: cp_type = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Risk", use_container_width=True, type="primary")

    if predict_btn:
        # Build raw row matching the training one-hot schema
        raw = {
            "Age": age, "Cholesterol": chol, "Blood_Pressure": bp,
            "Heart_Rate": hr, "Exercise_Hours": exercise, "Stress_Level": stress,
            "Blood_Sugar": sugar,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Smoking_Current": 1 if smoking == "Current" else 0,
            "Smoking_Former":  1 if smoking == "Former"  else 0,
            "Alcohol_Intake_Heavy":    1 if alcohol == "Heavy"    else 0,
            "Alcohol_Intake_Moderate": 1 if alcohol == "Moderate" else 0,
            "Family_History_Yes": 1 if family  == "Yes" else 0,
            "Diabetes_Yes":       1 if diabetes == "Yes" else 0,
            "Obesity_Yes":        1 if obesity  == "Yes" else 0,
            "Exercise_Induced_Angina_Yes": 1 if angina == "Yes" else 0,
            "Chest_Pain_Type_Non-Anginal Pain": 1 if cp_type == "Non-Anginal Pain" else 0,
            "Chest_Pain_Type_Typical Angina":   1 if cp_type == "Typical Angina"   else 0,
            "Chest_Pain_Type_Atypical Angina":  1 if cp_type == "Atypical Angina"  else 0,
        }
        input_df = pd.DataFrame([raw]).reindex(columns=MODEL_COLS, fill_value=0)

        clf  = MODELS[selected_model]
        prob = clf.predict_proba(input_df)[0][1]
        pred = int(prob > 0.5)
        label = "HIGH RISK ⚠️" if pred == 1 else "LOW RISK ✅"

        # ── Risk Gauge ────────────────────────────────────────────────────────
        g1, g2 = st.columns([1, 1])
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "Heart Disease Risk %", "font": {"size": 18, "color": "#e8e8f0"}},
                number={"suffix": "%", "font": {"size": 36, "color": "#fff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#888"},
                    "bar": {"color": "#ff4d6d" if prob > 0.5 else "#4dffb4"},
                    "steps": [
                        {"range": [0,  30],  "color": "#0d3320"},
                        {"range": [30, 60],  "color": "#3a3000"},
                        {"range": [60, 100], "color": "#3a0010"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0", height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            risk_color = "#ff4d6d" if pred == 1 else "#4dffb4"
            st.markdown(f"""
            <div class='metric-card'>
                <p>Prediction Result</p>
                <h2 class='{"risk-high" if pred==1 else "risk-low"}'>{label}</h2>
                <p>Probability: <strong style='color:{risk_color}'>{prob:.1%}</strong></p>
                <p>Model: <em>{selected_model}</em></p>
            </div>""", unsafe_allow_html=True)

            advice = (
                "🚨 Please consult a cardiologist. Reduce salt, exercise regularly, quit smoking."
                if pred == 1 else
                "✅ Keep up your healthy habits! Regular check-ups are still recommended."
            )
            st.info(advice)

        # ── SHAP Explainability ───────────────────────────────────────────────
        st.markdown("### 🔬 SHAP Explainability — Why this prediction?")
        try:
            import shap
            clf_inner = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf
            explainer = shap.TreeExplainer(clf_inner) if hasattr(clf_inner, "feature_importances_") \
                        else shap.Explainer(clf_inner, input_df)
            sv = explainer(input_df)
            shap_vals_arr = sv.values[0] if hasattr(sv, "values") else sv[0]
            if isinstance(shap_vals_arr, np.ndarray) and shap_vals_arr.ndim == 2:
                shap_vals_arr = shap_vals_arr[:, 1]

            shap_df = pd.DataFrame({
                "Feature":    MODEL_COLS,
                "SHAP Value": shap_vals_arr,
            }).assign(Direction=lambda d: d["SHAP Value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")) \
              .reindex(columns=["Feature","SHAP Value","Direction"]) \
              .sort_values("SHAP Value", key=abs, ascending=True).tail(12)

            fig_shap = px.bar(
                shap_df, x="SHAP Value", y="Feature", orientation="h",
                color="Direction",
                color_discrete_map={"Increases Risk": "#ff4d6d", "Decreases Risk": "#4dffb4"},
                title="Feature Contributions to This Prediction",
                template="plotly_dark",
            )
            fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig_shap, use_container_width=True)
            shap_dict = dict(zip(MODEL_COLS, map(float, shap_vals_arr)))
        except Exception as ex:
            st.warning(f"SHAP not available ({ex}). Showing global feature importance instead.")
            clf_inner = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf
            shap_dict = None
            if hasattr(clf_inner, "feature_importances_"):
                imp_df = pd.DataFrame({"Feature": MODEL_COLS, "Importance": clf_inner.feature_importances_}) \
                           .sort_values("Importance", ascending=True).tail(12)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                 title="Global Feature Importance", color="Importance",
                                 color_continuous_scale="Reds", template="plotly_dark")
                fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig_imp, use_container_width=True)

        # ── Save to history ───────────────────────────────────────────────────
        from datetime import datetime
        record = {
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Date":       datetime.now().strftime("%d/%m/%Y"),
            "Model":      selected_model,
            "Age":        age, "Gender": gender,
            "Cholesterol":chol, "BP": bp, "HR": hr,
            "Prob %":     round(prob * 100, 1),
            "Risk":       "High" if pred == 1 else "Low",
        }
        st.session_state.history.append(record)

        # ── PDF Download ──────────────────────────────────────────────────────
        patient_display = {
            "Age": age, "Gender": gender, "Cholesterol (mg/dL)": chol,
            "Blood Pressure (mmHg)": bp, "Heart Rate (bpm)": hr,
            "Blood Sugar (mg/dL)": sugar, "Smoking": smoking,
            "Alcohol": alcohol, "Exercise (hrs/wk)": exercise,
            "Stress Level": stress, "Family History": family,
            "Diabetes": diabetes, "Obesity": obesity,
            "Chest Pain Type": cp_type,
        }
        pdf_bytes = generate_pdf_report(patient_display, prob, label, shap_dict)
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name=f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("📊 Dataset Explorer")
    df = DF_RAW.copy()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Records",  f"{len(df):,}")
    kpi2.metric("Features",        "15")
    kpi3.metric("Heart Disease %", f"{df['Heart Disease'].mean():.1%}")
    kpi4.metric("Avg Age",         f"{df['Age'].mean():.0f} yrs")

    st.markdown("---")
    e1, e2 = st.columns(2)

    with e1:
        fig_dist = px.histogram(df, x="Age", color="Heart Disease",
            color_discrete_map={0:"#4dffb4",1:"#ff4d6d"}, barmode="overlay",
            title="Age Distribution by Heart Disease", template="plotly_dark",
            labels={"Heart Disease":"Heart Disease (1=Yes)"})
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with e2:
        pie = df["Heart Disease"].value_counts().rename({0:"No Disease",1:"Heart Disease"})
        fig_pie = px.pie(names=pie.index, values=pie.values,
            color_discrete_sequence=["#4dffb4","#ff4d6d"],
            title="Class Balance", template="plotly_dark", hole=0.4)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    e3, e4 = st.columns(2)
    with e3:
        fig_box = px.box(df, x="Heart Disease", y="Cholesterol", color="Heart Disease",
            color_discrete_map={0:"#4dffb4",1:"#ff4d6d"},
            title="Cholesterol vs Heart Disease", template="plotly_dark")
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

    with e4:
        fig_bp = px.box(df, x="Smoking", y="Blood Pressure", color="Heart Disease",
            color_discrete_map={0:"#4dffb4",1:"#ff4d6d"},
            title="Blood Pressure by Smoking Status", template="plotly_dark")
        fig_bp.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bp, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", template="plotly_dark",
        title="Feature Correlation Matrix")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### 📋 Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🏆 Model Performance Comparison")
    if METRICS:
        metrics_df = pd.DataFrame(METRICS).T.reset_index().rename(columns={"index": "Model"})

        # Radar chart
        categories = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
        fig_radar = go.Figure()
        colors = ["#4d9fff", "#ff4d6d", "#4dffb4"]
        for i, row in metrics_df.iterrows():
            vals = [row[c] for c in categories] + [row[categories[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=row["Model"],
                line_color=colors[i % len(colors)], opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0.5, 1], visible=True, color="#888")),
            showlegend=True, paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0", title="Model Metrics Radar",
            template="plotly_dark", height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Grouped bar chart
        melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_bar = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_sequence=colors, template="plotly_dark",
            title="Side-by-Side Metric Comparison")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.05])
        st.plotly_chart(fig_bar, use_container_width=True)

        # Table
        st.markdown("### 📊 Detailed Metrics Table")
        styled = metrics_df.set_index("Model").style \
            .background_gradient(cmap="Greens") \
            .format("{:.4f}")
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("No metrics found. Run `src/train_model.py` to generate them.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📋 Prediction History (This Session)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        # Risk trend
        if len(hist_df) > 1:
            fig_trend = px.line(hist_df, y="Prob %", markers=True,
                title="Risk Probability Trend Across Predictions",
                template="plotly_dark", color_discrete_sequence=["#ff4d6d"])
            fig_trend.add_hline(y=50, line_dash="dash", line_color="#888",
                                annotation_text="Risk Threshold (50%)")
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

        st.dataframe(hist_df, use_container_width=True)

        # Download
        csv_bytes = hist_df.to_csv(index=False).encode()
        st.download_button("📥 Download History (CSV)", csv_bytes,
                           "cardioai_history.csv", "text/csv",
                           use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the **🔬 Prediction** tab and analyse a patient!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🤖 CardioBot — Your AI Heart Health Assistant")
    st.caption("Ask me anything about heart disease, symptoms, lifestyle, or your results.")

    # Conversation display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""<div class='chat-msg-bot'>
            👋 Hi! I'm <strong>CardioBot</strong>, your AI heart health companion.<br>
            Ask me about cholesterol, blood pressure, symptoms, diet, exercise, or risk scores!
            </div>""", unsafe_allow_html=True)
        for turn in st.session_state.chat_history:
            st.markdown(f"<div class='chat-msg-user'>🧑 {turn['user']}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='chat-msg-bot'>🤖 {turn['bot']}</div>",
                        unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Your question:", placeholder="e.g. What does high cholesterol mean?")
        col_send, col_clear = st.columns([4, 1])
        with col_send:  send_btn  = st.form_submit_button("Send ➤", use_container_width=True)
        with col_clear: clear_btn = st.form_submit_button("Clear 🗑️")

    if send_btn and user_msg.strip():
        with st.spinner("CardioBot is thinking..."):
            response = get_chatbot_response(user_msg, st.session_state.chat_history)
        st.session_state.chat_history.append({"user": user_msg, "bot": response})
        st.rerun()

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    # Quick questions
    st.markdown("**💡 Quick Questions:**")
    qs = ["What is normal cholesterol?", "How to reduce heart disease risk?",
          "What are symptoms of a heart attack?", "How does stress affect the heart?"]
    qc = st.columns(len(qs))
    for i, q in enumerate(qs):
        with qc[i]:
            if st.button(q, key=f"quick_{i}"):
                resp = get_chatbot_response(q, st.session_state.chat_history)
                st.session_state.chat_history.append({"user": q, "bot": resp})
                st.rerun()
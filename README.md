# 🫀 CardioAI: Heart Disease Risk Prediction & Analysis Platform

CardioAI is a full-featured, interactive platform designed to assess, analyze, and explain heart disease risk using advanced Machine Learning. It consists of a predictive REST API, a dynamic Streamlit frontend, multi-model evaluation metrics, exploratory data analysis tools, and an AI Chatbot companion.

---

## 🌟 Key Features

### 🔬 Multi-Model Risk Prediction
- Predict risk using optimized machine learning models:
  - **Random Forest** (Highly interpretable, feature-importance driven)
  - **XGBoost** (Optimized gradient boosted decision trees)
  - **Logistic Regression** (Linear baseline)
- Interactive sliders and metrics inputs to customize demographics, clinical tests, and lifestyle patterns.

### 📊 Exploratory Data Analysis (EDA) Explorer
- Live distribution charts of age by heart disease status.
- Class balance visualizer.
- Feature vs Target correlation heatmaps and boxplots.

### 🏆 Model Comparison
- Dynamic metrics radar charts comparing **Accuracy, Precision, Recall, F1-Score**, and **AUC-ROC**.
- Clear side-by-side performance bar plots and a detailed metric tracking table.

### 🔬 Advanced Explainability (SHAP)
- Real-time prediction explanations with local SHAP values showing exactly why a model flags a patient as high/low risk.

### 📋 Patient History & Dynamic Reporting
- Keep a session-based history of all patient evaluations.
- Export findings into a highly detailed **PDF Report** or download all history as a CSV.

### 🤖 CardioBot — AI Chatbot Assistant
- AI-driven chat assistant to answer health questions, explain medical terminology, and provide preventative lifestyle advice.

---

## 🏗️ Project Architecture

```
Heart_Disease/
├── api/
│   └── main.py             # FastAPI REST API implementation
├── app/
│   ├── app.py              # Main Streamlit Dashboard UI
│   ├── chatbot.py          # AI Chatbot logic & prompts
│   └── report_generator.py # PDF Report generation using ReportLab
├── data/
│   └── heart.csv           # Base training dataset
├── models/
│   ├── *_model.pkl         # Trained scikit-learn & xgboost models
│   ├── model_columns.pkl   # Training feature columns
│   └── model_metrics.json  # Comprehensive cross-model evaluation stats
├── notebooks/
│   └── exploration.ipynb   # Model training & EDA experiments
├── src/
│   ├── predict.py          # Command-line prediction script
│   └── train_model.py      # Production training & metrics generation
├── requirements.txt        # Full project dependencies
└── .gitignore              # Git ignore rules
```

---

## 🚀 Quick Start & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/WithSamir/Heart_Disease.git
cd Heart_Disease
```

### 2️⃣ Install Dependencies
It's recommended to set up a virtual environment first:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Retrain Models (Optional)
If you'd like to refresh models and metrics, execute the training pipeline:
```bash
python3 src/train_model.py
```

---

## 🖥️ Launching the Application

### Running the FastAPI REST API
To spin up the CardioAI prediction backend API:
```bash
uvicorn api.main:app --reload
```
Once running, check out the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Running the Streamlit Dashboard
To run the fully-featured frontend with Prediction, EDA, and Chatbot tabs:
```bash
streamlit run app/app.py
```
Open your browser at the local URL returned by Streamlit (e.g., [http://localhost:8501](http://localhost:8501)).

---

## 📄 License
This project is for educational and research purposes only. Not intended for use as a clinical diagnostic tool.

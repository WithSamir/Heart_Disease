import joblib
import pandas as pd

# load model and columns
model = joblib.load("models/heart_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# example input
input_data = {
    "Age": 52,
    "Cholesterol": 230,
    "Blood_Pressure": 130,
    "Heart_Rate": 150,
    "Exercise_Hours": 3,
    "Stress_Level": 5,
    "Blood_Sugar": 110
}

df = pd.DataFrame([input_data])

# match training columns
df = df.reindex(columns=model_columns, fill_value=0)

prediction = model.predict(df)

if prediction[0] == 1:
    print("High Risk of Heart Disease")
else:
    print("Low Risk of Heart Disease")
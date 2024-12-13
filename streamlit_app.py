import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('diabetes.csv')

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Streamlit app layout
st.title("Diabetes Risk Prediction App")

# User input for all features
pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 31.4)
diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
age = st.slider("Age (years)", 21, 81, 33)

# Prepare input for prediction
user_input = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]])

# Scale the input
user_input_scaled = scaler.transform(user_input)

# Predictions
log_reg_pred = log_reg.predict_proba(user_input_scaled)[:, 1][0]
rf_pred = rf.predict_proba(user_input_scaled)[:, 1][0]

# Display results
st.write("### Prediction Results")
st.write(f"*Logistic Regression Probability of Diabetes:* {log_reg_pred:.2f}")
st.write(f"*Random Forest Probability of Diabetes:* {rf_pred:.2f}")

# Recommendations
if log_reg_pred > 0.5 or rf_pred > 0.5:
    st.write("*Recommendation:* Consult a healthcare provider.")
else:
    st.write("*Recommendation:* Your risk of diabetes seems low. Maintain a healthy lifestyle.")

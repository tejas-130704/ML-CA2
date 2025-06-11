import streamlit as st
import numpy as np
import joblib

# Load model, encoders, scaler
model = joblib.load("model/elastic_grid.pkl")
scaler = joblib.load("model/scaler.pkl")
le_sex = joblib.load("model/le_sex.pkl")
le_day = joblib.load("model/le_day.pkl")
le_time = joblib.load("model/le_time.pkl")

st.title("Waiter's Tip Predictor")

# User Inputs
total_bill = st.number_input("Total Bill", min_value=0.0, step=0.01)
sex = st.selectbox("Sex", options=le_sex.classes_)
# day = st.selectbox("Day", options=le_day.classes_)
time = st.selectbox("Time", options=le_time.classes_)
size = st.slider("Size of the Group", min_value=1, max_value=10)

# Encode & scale input
input_encoded = np.array([
    total_bill,
    le_sex.transform([sex])[0],
    # le_day.transform([day])[0],
    le_time.transform([time])[0],
    size
]).reshape(1, -1)

input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Tip"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Tip Amount: ${prediction[0]:.2f}")

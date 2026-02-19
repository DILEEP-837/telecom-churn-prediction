import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("telecom_churn.csv")

# Train model
X = data[["Age", "MonthlyRecharge", "Tenure", "RechargeFrequency"]]
y = data["Churn"]

model = LogisticRegression()
model.fit(X, y)

# App title
st.title("Telecom Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
recharge = st.number_input("Enter Monthly Recharge Amount", min_value=0, value=100)
tenure = st.number_input("Enter Tenure (months)", min_value=0, value=12)
frequency = st.number_input("Enter Recharge Frequency", min_value=0, value=5)

# Predict button
if st.button("Predict"):

    prediction = model.predict([[age, recharge, tenure, frequency]])

    if prediction[0] == 0:
        st.success("Customer will continue recharging")
    else:
        st.error("Customer will stop recharging")

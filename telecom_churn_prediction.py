import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("telecom_churn.csv")

# Input features
X = data[["Age", "MonthlyRecharge", "Tenure", "RechargeFrequency"]]

# Output
y = data["Churn"]

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict new customer
prediction = model.predict([[30, 49, 2, 2]])

# Show result
if prediction[0] == 0:
    print("Customer will continue recharging")
else:
    print("Customer will stop recharging")

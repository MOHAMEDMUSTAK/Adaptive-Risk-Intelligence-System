import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
fraud_model = pickle.load(open("fraud_model.pkl", "rb"))
anomaly_model = pickle.load(open("anomaly_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.set_page_config(page_title="Financial Risk Intelligence Engine", layout="centered")
st.title("Financial Risk Intelligence Engine")
st.write("This system analyzes whether a new transaction looks normal or suspicious based on past behavior patterns.")
# -----------------------------
# Simulated User Spending History
# -----------------------------
np.random.seed(42)
history_size = 20
transaction_history = np.random.normal(250, 60, history_size)
avg_spending = np.mean(transaction_history)
st.subheader("Recent Spending Pattern")
st.write("Below is a simulated recent spending pattern of the user.")
st.line_chart(transaction_history)
st.write(f"Average Spending: ₹{round(avg_spending, 2)}")
# -----------------------------
# User-Friendly Inputs
# -----------------------------
st.subheader("New Transaction Details")
amount = st.number_input(
    "How much is the new transaction amount? (₹)",
    min_value=0.0,
    max_value=100000.0,
    value=250.0,
    help="Enter the amount of the new transaction"
)
location_choice = st.radio(
    "Where is this transaction happening?",
    ["Same country as usual", "Different / Foreign country"]
)
device_choice = st.radio(
    "Which device is being used?",
    ["Regular personal device", "New / Unknown device"]
)
# Convert to model-friendly format
location_risk = 1 if location_choice == "Different / Foreign country" else 0
device_risk = 1 if device_choice == "New / Unknown device" else 0
# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Transaction Risk"):

    transaction_frequency = history_size
    spending_spike = amount > (avg_spending * 1.8)

    input_data = np.array([[amount, transaction_frequency, location_risk, device_risk]])
    scaled = scaler.transform(input_data)

    fraud_prob = fraud_model.predict_proba(scaled)[0][1]
    anomaly_score = anomaly_model.decision_function(scaled)[0]
    behavioral_penalty = 0.15 if spending_spike else 0

    final_risk = (fraud_prob * 0.6) + ((1 - anomaly_score) * 0.4) + behavioral_penalty
    final_risk = min(final_risk, 1)

    risk_score = round(final_risk * 100, 2)

    st.subheader("Risk Analysis Result")

    st.write(f"Fraud Probability: {round(fraud_prob * 100, 2)}%")
    st.write(f"Final Risk Score: {risk_score}%")

    if spending_spike:
        st.warning("Unusual spending detected compared to past behavior.")

    if risk_score > 70:
        st.error("High Risk Transaction — Manual verification recommended.")
    elif risk_score > 40:
        st.warning("Moderate Risk — Additional verification may be required.")
    else:
        st.success("Low Risk — Transaction appears normal.")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Fraud Probability", "Final Risk Score"],
           [fraud_prob * 100, risk_score])
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

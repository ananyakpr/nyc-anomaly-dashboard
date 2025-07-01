import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("🚕 NYC Ride Demand Anomaly Detection")
st.write("Real-time anomaly detection in hourly ride counts")

# Load model and data
model = joblib.load("ride_anomaly_model.pkl")
df = pd.read_csv("labeled_ride_counts.csv")
df['hour'] = pd.to_datetime(df['hour'])

# Predict anomalies
df['anomaly'] = model.predict(df[['ride_count']])

# Plotting
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['hour'], df['ride_count'], label='Ride Count', color='blue')
anomalies = df[df['anomaly'] == -1]
ax.scatter(anomalies['hour'], anomalies['ride_count'], color='red', label='Anomaly')
ax.set_xlabel("Hour")
ax.set_ylabel("Ride Count")
ax.set_title("Ride Demand with Anomalies")
ax.legend()
st.pyplot(fig)

st.write(f"🚨 Total anomalies detected: {len(anomalies)}")

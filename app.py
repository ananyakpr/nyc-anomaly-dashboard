import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

# Page config
st.set_page_config(page_title="NYC Ride Anomaly Stream", layout="wide")
st.markdown("<h1 style='color: white;'>🚦 NYC Ride Demand - Real-Time Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("Live simulation of ride demand anomalies using Isolation Forest model.")

# Load model and data
model = joblib.load("ride_anomaly_model.pkl")
df = pd.read_csv("labeled_ride_counts.csv")

# ✅ Use the correct datetime column
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

# Streamlit placeholders
chart_area = st.empty()
status_area = st.empty()

# Lists to store stream values
ride_counts = []
timestamps = []
anomaly_points = []

# Simulate streaming
for i in range(len(df)):
    ts = df.loc[i, "hour"]
    rc = df.loc[i, "ride_count"]

    timestamps.append(ts)
    ride_counts.append(rc)

    # Predict anomaly
    pred = model.predict([[rc]])
    if pred[0] == -1:
        anomaly_points.append((ts, rc))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(timestamps, ride_counts, color='blue', label='Ride Count')
    if anomaly_points:
        x, y = zip(*anomaly_points)
        ax.scatter(x, y, color='red', label='Anomaly')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Ride Count")
    ax.set_title("Live Ride Demand with Anomalies")
    ax.legend()

    chart_area.pyplot(fig)
    status_area.info(f"🕐 {ts} | 🚕 Ride Count: {rc} | {'🔴 Anomaly' if pred[0] == -1 else '✅ Normal'}")

    time.sleep(0.5)

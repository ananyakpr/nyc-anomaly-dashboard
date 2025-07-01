# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="NYC Live Ride Anomaly Detector", layout="wide")
st.title("🚖 NYC Live Simulated Ride Anomaly Detector")
st.caption("Simulated real-time ride data • Powered by Isolation Forest")

# Load your trained model
model = joblib.load("ride_anomaly_model.pkl")

# Initialize session state for live data
if "data" not in st.session_state:
    now = datetime.utcnow()
    st.session_state.data = pd.DataFrame([
        {
            "hour": now - timedelta(hours=i),
            "ride_count": random.randint(11000, 17000)
        }
        for i in range(6)
    ])

# Update data with new simulated point
def update_data():
    now = datetime.utcnow()
    ride_count = random.randint(11000, 17000)
    if random.random() < 0.15:
        ride_count = random.choice([random.randint(3000, 4000), random.randint(22000, 25000)])

    new_point = pd.DataFrame([{
        "hour": now,
        "ride_count": ride_count
    }])

    df = pd.concat([st.session_state.data, new_point], ignore_index=True)
    df = df.sort_values("hour").tail(10)  # Keep last 10 points
    st.session_state.data = df

# Live update checkbox
auto = st.checkbox("🔁 Auto-refresh every 5 seconds", value=False)

# Manual refresh button
if st.button("🔄 Generate New Ride Data"):
    update_data()

# Auto-refresh mode
if auto:
    while True:
        update_data()
        time.sleep(5)
        st.rerun()

# Run anomaly detection
df = st.session_state.data.copy()
df["anomaly"] = model.predict(df[["ride_count"]])

# Plot
fig, ax = plt.subplots()
ax.plot(df["hour"], df["ride_count"], label="Ride Count", color="royalblue")
anomalies = df[df["anomaly"] == -1]
ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")

ax.set_title("Simulated NYC Ride Demand (Live Stream)")
ax.set_xlabel("Time")
ax.set_ylabel("Ride Count")
ax.legend()
st.pyplot(fig)

# Latest status
latest = df.iloc[-1]
if latest["anomaly"] == -1:
    st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly")
else:
    st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

# Show data
with st.expander("📊 Raw Ride Data"):
    st.dataframe(df)

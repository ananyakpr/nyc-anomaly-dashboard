# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Ride Anomalies", layout="wide")

st.title("🚖 NYC Simulated Ride Anomaly Detector")
st.caption("Simulated ride data • Powered by Isolation Forest")

# Load trained model
model = joblib.load("ride_anomaly_model.pkl")

# Step A: Generate simulated ride data
@st.cache_data(ttl=600)
def simulate_ride_data():
    now = datetime.utcnow()
    data = []

    for i in range(6):
        hour = now - timedelta(hours=5 - i)
        ride_count = random.randint(11000, 17000)

        # Inject anomaly with 10% probability
        if random.random() < 0.15:
            ride_count = random.choice([random.randint(3000, 4000), random.randint(22000, 25000)])

        data.append({
            "hour": hour,
            "ride_count": ride_count
        })

    return pd.DataFrame(data)

# Get simulated data
df = simulate_ride_data()

# Step B: Predict anomalies
df["anomaly"] = model.predict(df[["ride_count"]])

# Step C: Plot results
fig, ax = plt.subplots()
ax.plot(df["hour"], df["ride_count"], label="Ride Count", color="royalblue")

# Highlight anomalies
anomalies = df[df["anomaly"] == -1]
ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")

ax.set_title("Simulated NYC Taxi Ride Demand - Last 6 Hours")
ax.set_xlabel("Time")
ax.set_ylabel("Ride Count")
ax.legend()
st.pyplot(fig)

# Step D: Display result of latest hour
latest = df.iloc[-1]
if latest["anomaly"] == -1:
    st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly")
else:
    st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

# Optional: Show full data
with st.expander("See full simulated dataset"):
    st.dataframe(df)

# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="NYC Ride Stream", layout="wide")
st.title("🚖 NYC Ride Demand - Real-Time Stream")
st.caption("Simulated live stream • Anomaly detection using Isolation Forest")

# Load trained model
model = joblib.load("ride_anomaly_model.pkl")

# Initialize session state
if "stream_data" not in st.session_state:
    now = datetime.utcnow()
    st.session_state.stream_data = pd.DataFrame([
        {
            "hour": now - timedelta(hours=i),
            "ride_count": random.randint(11000, 17000)
        }
        for i in range(10)
    ])
    st.session_state.stream_data = st.session_state.stream_data.sort_values("hour")

placeholder = st.empty()

# Continuous loop
for _ in range(100):  # runs for 100 cycles, you can increase it
    # Simulate new data
    now = datetime.utcnow()
    ride_count = random.randint(11000, 17000)
    if random.random() < 0.15:  # inject anomaly
        ride_count = random.choice([random.randint(3000, 5000), random.randint(22000, 25000)])

    new_point = pd.DataFrame([{
        "hour": now,
        "ride_count": ride_count
    }])

    # Update the stream
    st.session_state.stream_data = pd.concat(
        [st.session_state.stream_data, new_point], ignore_index=True
    ).sort_values("hour").tail(10)  # keep last 10 entries

    df = st.session_state.stream_data.copy()
    df["anomaly"] = model.predict(df[["ride_count"]])

    # Draw the chart
    with placeholder.container():
        fig, ax = plt.subplots()
        ax.plot(df["hour"], df["ride_count"], label="Ride Count", color="royalblue")
        anomalies = df[df["anomaly"] == -1]
        ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")
        ax.set_title("Live NYC Ride Stream")
        ax.set_xlabel("Time")
        ax.set_ylabel("Ride Count")
        ax.legend()
        st.pyplot(fig)

        latest = df.iloc[-1]
        if latest["anomaly"] == -1:
            st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly")
        else:
            st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

        with st.expander("📊 Raw Stream Data"):
            st.dataframe(df.reset_index(drop=True))

    time.sleep(5)
    st.rerun()

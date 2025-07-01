import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

# 🎨 Page config
st.set_page_config(page_title="NYC Ride Demand Anomalies", layout="wide")

# 📚 Sidebar info
st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**NYC Ride Demand Anomaly Detection**  
Built by *Ananya Kapoor*  
Live simulation of hourly taxi rides in NYC, with real-time anomaly detection using Isolation Forest.

- Model: Isolation Forest  
- Data: Historical ride demand  
- Purpose: Spot unusual demand (e.g., strikes, weather spikes)
""")

# 🧠 Load model and data
model = joblib.load("ride_anomaly_model.pkl")
df = pd.read_csv("labeled_ride_counts.csv")

df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

# 🔄 Simulation state
ride_counts = []
timestamps = []
anomaly_points = []

# 🎯 Main title
st.markdown("<h1 style='text-align: center; color: white;'>🚦 NYC Ride Demand - Live Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

chart_area = st.empty()
status_box = st.empty()

# 🌀 Streaming loop
for i in range(len(df)):
    ts = df.loc[i, "hour"]
    rc = df.loc[i, "ride_count"]

    timestamps.append(ts)
    ride_counts.append(rc)

    pred = model.predict([[rc]])
    if pred[0] == -1:
        anomaly_points.append((ts, rc))

    # 📈 Plot
    fig, ax = plt.subplots()
    ax.plot(timestamps, ride_counts, color='skyblue', label='Ride Count')
    if anomaly_points:
        x, y = zip(*anomaly_points)
        ax.scatter(x, y, color='red', label='Anomaly', zorder=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Ride Count")
    ax.set_title("📈 Ride Demand Over Time")
    ax.legend()

    chart_area.pyplot(fig)

    # ✅ / 🔴 Status display
    if pred[0] == -1:
        status_box.error(f"🔴 {ts} | Ride Count: {rc} → Anomaly Detected!")
    else:
        status_box.success(f"✅ {ts} | Ride Count: {rc} → Normal")

    time.sleep(0.5)

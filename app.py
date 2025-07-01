import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Real-Time Taxi Anomalies", layout="wide")

# Sidebar info
st.sidebar.title("📊 Real-Time Ride Demand")
st.sidebar.markdown("""
Live anomaly detection on NYC taxi data via public API.

- 🔄 Refreshes latest hour
- 🧠 ML Model: Isolation Forest
- 📍 Source: NYC OpenData API
""")

# Load model
model = joblib.load("ride_anomaly_model.pkl")

# Function to fetch data from NYC TLC API
@st.cache_data(ttl=3600)
def fetch_latest_ride_data():
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)
    url = "https://data.cityofnewyork.us/resource/2yzn-sicd.json"
    params = {
        "$where": f"pickup_datetime between '{last_hour.strftime('%Y-%m-%dT%H:%M:%S')}' and '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "$limit": 50000
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    return df

# Get and process data
df = fetch_latest_ride_data()

if df.empty or "pickup_datetime" not in df.columns:
    st.error("🚨 No ride data retrieved for the past hour. Please try again later.")
else:
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.floor("H")
    hourly_counts = df.groupby("hour").size().reset_index(name="ride_count")
    hourly_counts = hourly_counts.sort_values("hour").reset_index(drop=True)

    # Detect anomalies
    hourly_counts["anomaly"] = model.predict(hourly_counts[["ride_count"]])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(hourly_counts["hour"], hourly_counts["ride_count"], label="Ride Count", color="skyblue")

    anomalies = hourly_counts[hourly_counts["anomaly"] == -1]
    if not anomalies.empty:
        ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Ride Count")
    ax.set_title("NYC Taxi Ride Count - Real-Time Anomaly Detection")
    ax.legend()
    st.pyplot(fig)

    # Show status for last timestamp
    latest = hourly_counts.iloc[-1]
    if latest["anomaly"] == -1:
        st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly Detected")
    else:
        st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

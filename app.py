import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Ride Anomalies", layout="wide")

st.title("🚖 NYC Real-Time Ride Anomaly Detector")
st.caption("Live from NYC OpenData API • Powered by Isolation Forest")

# Load your trained model
model = joblib.load("ride_anomaly_model.pkl")

# 🧠 Cache API response for 1 hour to save memory
@st.cache_data(ttl=3600)
def fetch_latest_data():
    now = datetime.utcnow()
    past = now - timedelta(hours=6)  # only last 6 hours
    url = "https://data.cityofnewyork.us/resource/2yzn-sicd.json"
    params = {
        "$where": f"pickup_datetime between '{past.strftime('%Y-%m-%dT%H:%M:%S')}' and '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "$limit": 5000  # LIMIT memory usage
    }
    response = requests.get(url, params=params)
    return pd.DataFrame(response.json())

# 🔁 Fetch and inspect data
df = fetch_latest_data()

# 🔎 Show what the API returned (TEMPORARY debug line)
st.write("First few rows from API:", df.head())
st.write("Available columns:", df.columns)

# ✅ Check for correct timestamp column
pickup_col = None
for col in ["pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"]:
    if col in df.columns:
        pickup_col = col
        break

if not pickup_col:
    st.error("🚨 Could not find pickup time column in API data.")
    st.stop()

# 🧹 Clean and group
df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
df = df.dropna(subset=[pickup_col])
df["hour"] = df[pickup_col].dt.floor("H")
hourly = df.groupby("hour").size().reset_index(name="ride_count")


if df.empty or "pickup_datetime" not in df.columns:
    st.error("🚨 No valid data from API.")
    st.stop()

# 🧹 Clean and aggregate
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna(subset=["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.floor("H")
hourly = df.groupby("hour").size().reset_index(name="ride_count")

if hourly.shape[0] < 4:
    st.warning("Not enough data to analyze. Try again later.")
    st.stop()

# 🧠 Predict anomalies
hourly["anomaly"] = model.predict(hourly[["ride_count"]])

# 📊 Plot
fig, ax = plt.subplots()
ax.plot(hourly["hour"], hourly["ride_count"], label="Ride Count", color="royalblue")

anomalies = hourly[hourly["anomaly"] == -1]
ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")

ax.set_title("NYC Taxi Ride Demand - Last 6 Hours")
ax.set_xlabel("Time")
ax.set_ylabel("Ride Count")
ax.legend()

st.pyplot(fig)

# ✅ Status message
latest = hourly.iloc[-1]
if latest["anomaly"] == -1:
    st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly")
else:
    st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

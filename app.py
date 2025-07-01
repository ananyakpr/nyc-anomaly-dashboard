import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime

# Configure Streamlit page
st.set_page_config(page_title="NYC Ride Anomalies", layout="wide")
st.title("🚖 NYC Ride Anomaly Detector")
st.caption("Simulated Ride Stream • Powered by Isolation Forest")

# Load the trained model
model = joblib.load("ride_anomaly_model.pkl")

# Fetch data from FastAPI backend
API_URL = "http://localhost:8010/rides"
response = requests.get(API_URL)

if response.status_code != 200:
    st.error("🚨 Failed to fetch simulated ride data from API.")
    st.stop()

# Convert response to DataFrame
data = response.json()
df = pd.DataFrame(data)

# Debug output
st.subheader("📦 Incoming Data Preview")
st.write(df.head())

# Check and process timestamp column
if "hour" not in df.columns or "ride_count" not in df.columns:
    st.error("❌ Required columns not found in the API response.")
    st.stop()

df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
df = df.dropna(subset=["hour"])
df = df.sort_values("hour")

# Predict anomalies
df["anomaly"] = model.predict(df[["ride_count"]])

# Plot the results
fig, ax = plt.subplots()
ax.plot(df["hour"], df["ride_count"], label="Ride Count", color="royalblue")
anomalies = df[df["anomaly"] == -1]
ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")

ax.set_title("Simulated NYC Taxi Ride Demand - Last 6 Hours")
ax.set_xlabel("Time")
ax.set_ylabel("Ride Count")
ax.legend()

st.pyplot(fig)

# Show latest point status
latest = df.iloc[-1]
if latest["anomaly"] == -1:
    st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly")
else:
    st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

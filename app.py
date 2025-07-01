import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Ride Anomalies", layout="wide")

st.title("🚖 NYC Real-Time Ride Anomaly Detector")
st.caption("Live from NYC OpenData API • Powered by Isolation Forest")

# Load trained model
model = joblib.load("ride_anomaly_model.pkl")

@st.cache_data(ttl=3600)
def fetch_latest_data():
    now = datetime.utcnow()
    past = now - timedelta(hours=6)
    url = "https://data.cityofnewyork.us/resource/2yzn-sicd.json"
    params = {
        "$where": f"pickup_datetime between '{past.strftime('%Y-%m-%dT%H:%M:%S')}' and '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "$limit": 5000
    }
    response = requests.get(url, params=params)
    return pd.DataFrame(response.json())

# Step 1: Fetch and inspect
df = fetch_latest_data()

# 🛠 Debug block
if df.empty:
    st.error("❌ API returned no data.")
    st.stop()


# Step 2: Detect the correct timestamp column
POSSIBLE_COLS = ["pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime", "timestamp", "trip_pickup_datetime"]
pickup_col = next((col for col in POSSIBLE_COLS if col in df.columns), None)

if pickup_col is None:
    st.error("❌ Could not detect timestamp column in API data.")
    st.write("🔍 Columns available:", df.columns.tolist())
    st.stop()

# Step 3: Clean and process
df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
df = df.dropna(subset=[pickup_col])
df["hour"] = df[pickup_col].dt.floor("H")
hourly = df.groupby("hour").size().reset_index(name="ride_count")

if hourly.shape[0] < 4:
    st.warning("⚠️ Not enough recent data to detect anomalies. Please check again later.")
    st.dataframe(df.head())
    st.stop()

# Step 4: Predict
hourly["anomaly"] = model.predict(hourly[["ride_count"]])

# Step 5: Plot
fig, ax = plt.subplots()
ax.plot(hourly["hour"], hourly["ride_count"], label="Ride Count", color="royalblue")
anomalies = hourly[hourly["anomaly"] == -1]
ax.scatter(anomalies["hour"], anomalies["ride_count"], color="red", label="Anomaly")
ax.set_title("NYC Taxi Ride Demand - Last 6 Hours")
ax.set_xlabel("Time")
ax.set_ylabel("Ride Count")
ax.legend()
st.pyplot(fig)

# Step 6: Status
latest = hourly.iloc[-1]
if latest["anomaly"] == -1:
    st.error(f"🔴 {latest['hour']} | Ride Count: {latest['ride_count']} → Anomaly Detected")
else:
    st.success(f"✅ {latest['hour']} | Ride Count: {latest['ride_count']} → Normal")

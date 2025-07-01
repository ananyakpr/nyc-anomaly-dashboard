# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Taxi Anomaly Detector", layout="wide")
st.title("🚖 NYC Real-Time Ride Anomaly Detector")
st.caption("Live from NYC OpenData • Powered by Isolation Forest")

# Load model
model = joblib.load("ride_anomaly_model.pkl")

# NYC Open Data API (Yellow Taxi Trip Data)
NYC_API = "https://data.cityofnewyork.us/resource/2yzn-sicd.json"

@st.cache_data(ttl=3600)
def fetch_data():
    now = datetime.utcnow()
    past = now - timedelta(hours=6)
    
    params = {
        "$where": f"pickup_datetime between '{past.strftime('%Y-%m-%dT%H:%M:%S')}' and '{now.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "$limit": 5000
    }

    response = requests.get(NYC_API, params=params)
    df = pd.DataFrame(response.json())
    return df

df = fetch_data()

# Detect timestamp column
timestamp_col = None
for col in ["pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"]:
    if col in df.columns:
        timestamp_col = col
        break

if not timestamp_col:
    st.error("❌ No pickup time column found in API response.")
    st.stop()

# Preprocessing
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df = df.dropna(subset=[timestamp_col])
df["hour"] = df[timestamp_col].dt.floor("H")

hourly = df.groupby("hour").size().reset_index(name="ride_count")

if hou

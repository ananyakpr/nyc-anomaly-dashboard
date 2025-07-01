# 🚖 NYC Live Ride Anomaly Detector

A real-time anomaly detection dashboard that simulates NYC taxi ride demand. Built using **Streamlit**, powered by an **Isolation Forest** machine learning model.

---

## 🔧 Features
- Simulated live ride data with automatic updates every 5 seconds
- Anomaly detection using Isolation Forest
- Real-time graph visualization
- Visual alerts for anomalies
- Interactive and minimal UI via Streamlit

---

## 🛠️ Tech Stack
- Python
- Streamlit
- scikit-learn
- pandas
- matplotlib
- joblib (for model serialization)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/nyc-anomaly-dashboard.git
cd nyc-anomaly-dashboard
pip install -r requirements.txt
streamlit run app.py

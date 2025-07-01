# backend/fake_api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random

app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/rides")
def get_fake_rides():
    now = datetime.utcnow()
    data = []

    for i in range(6):  # Last 6 hours
        hour = now - timedelta(hours=i)
        ride_count = random.randint(10000, 18000)

        if random.random() < 0.1:  # Inject anomaly
            ride_count = random.choice([random.randint(2500, 4000), random.randint(22000, 25000)])

        data.append({
            "hour": hour.strftime("%Y-%m-%dT%H:%M:%S"),
            "ride_count": ride_count
        })

    return list(reversed(data))

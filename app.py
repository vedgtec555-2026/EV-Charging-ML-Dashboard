
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import plotly.express as px

st.set_page_config(
    page_title="EV Charging AI System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart EV Charging Prediction & Recommendation System")
st.markdown("AI-based system for predicting EV charging slot availability and waiting time")

@st.cache_data
def create_data():

    n = 5000
    start = datetime(2026, 1, 1)

    cities = ["Mumbai", "Pune", "Nagpur", "Nashik"]
    weather_list = ["Sunny", "Rainy", "Cloudy"]

    data = []

    for i in range(n):

        time = start + timedelta(minutes=10*i)
        hour = time.hour

        total_ports = random.choice([4, 6, 8, 10])

        if 17 <= hour <= 22:
            occupied = random.randint(total_ports//2, total_ports)
            traffic = "High"

        elif 8 <= hour <= 11:
            occupied = random.randint(1, total_ports-1)
            traffic = "Medium"

        else:
            occupied = random.randint(0, total_ports//2)
            traffic = "Low"

        queue = 0

        if occupied >= total_ports:
            queue = random.randint(1, 4)

        wait_time = queue * random.randint(5, 15)

        available = 1 if occupied < total_ports else 0

        row = {
            "City": random.choice(cities),
            "Hour": hour,
            "Traffic": traffic,
            "Weather": random.choice(weather_list),
            "Total_Ports": total_ports,
            "Occupied_Ports": occupied,
            "Queue_Length": queue,
            "Battery_Level": random.randint(20, 100),
            "Availability": available,
            "Waiting_Time": wait_time
        }

        data.append(row)

    return pd.DataFrame(data)

df = create_data()

df_encoded = df.copy()

traffic_map = {"Low": 0, "Medium": 1, "High": 2}
weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2}
city_map = {"Mumbai": 0, "Pune": 1, "Nagpur": 2, "Nashik": 3}

df_encoded["Traffic"] = df_encoded["Traffic"].map(traffic_map)
df_encoded["Weather"] = df_encoded["Weather"].map(weather_map)
df_encoded["City"] = df_encoded["City"].map(city_map)

X = df_encoded.drop(["Availability", "Waiting_Time"], axis=1)

y_class = df_encoded["Availability"]
y_reg = df_encoded["Waiting_Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

clf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf_model.fit(X_train, y_train)

reg_model = LinearRegression()
reg_model.fit(X, y_reg)

st.sidebar.header("🔋 EV User Input")

city = st.sidebar.selectbox(
    "Select City",
    ["Mumbai", "Pune", "Nagpur", "Nashik"]
)

hour = st.sidebar.slider(
    "Select Hour",
    0,
    23,
    18
)

traffic = st.sidebar.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Sunny", "Rainy", "Cloudy"]
)

total_ports = st.sidebar.selectbox(
    "Total Charging Ports",
    [4, 6, 8, 10]
)

occupied_ports = st.sidebar.slider(
    "Occupied Ports",
    0,
    total_ports,
    total_ports // 2
)

queue = st.sidebar.slider(
    "Queue Length",
    0,
    5,
    1
)

battery = st.sidebar.slider(
    "Battery Demand",
    20,
    100,
    60
)

predict = st.sidebar.button("🔮 Predict")

if predict:

    input_data = pd.DataFrame([{
        "City": city_map[city],
        "Hour": hour,
        "Traffic": traffic_map[traffic],
        "Weather": weather_map[weather],
        "Total_Ports": total_ports,
        "Occupied_Ports": occupied_ports,
        "Queue_Length": queue,
        "Battery_Level": battery
    }])

    availability = clf_model.predict(input_data)[0]

    probability = clf_model.predict_proba(input_data)[0][1] * 100

    waiting_time = reg_model.predict(input_data)[0]

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Availability Probability",
            f"{probability:.2f}%"
        )

    with col2:
        st.metric(
            "Expected Waiting Time",
            f"{waiting_time:.2f} min"
        )

    with col3:
        if availability == 1:
            st.success("✅ Slot Available")
        else:
            st.error("❌ No Slot Available")

hourly = df.groupby("Hour")["Occupied_Ports"].mean().reset_index()

fig1 = px.line(
    hourly,
    x="Hour",
    y="Occupied_Ports",
    title="Hourly Charging Demand"
)

st.plotly_chart(fig1, use_container_width=True)

import streamlit as st
import pandas as pd
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AgriPredict Pro 🌾", layout="wide")

API_KEY = "64241a8292621154f54967271c80b896"

# -------------------------
# UI DESIGN
# -------------------------
st.markdown("""
<style>
.stApp {
    background: #020617;
    color: white;
}
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #22c55e;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}
.stButton>button {
    background: #22c55e;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌾 AgriPredict Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Based Smart Crop Recommendation System</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA (FIXED PATH)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

try:
    data = pd.read_csv(file_path)
except:
    st.error("❌ Dataset not found. Please check file upload.")
    st.stop()

# DEBUG (can remove later)
st.write("📂 Files available:", os.listdir(BASE_DIR))

# -------------------------
# MODEL TRAINING
# -------------------------
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------
# WEATHER FUNCTION
# -------------------------
def get_weather(city):
    city = city.strip().title()
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        res = requests.get(url)
        data = res.json()

        if data.get("cod") == 200:
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"]
            }
        else:
            return None
    except:
        return None

# -------------------------
# LAYOUT
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# INPUT SECTION
# -------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Inputs")

    city = st.text_input("Enter City (optional)")
    weather = None

    if city:
        weather = get_weather(city)

        if weather:
            st.success(f"🌤 Temp: {weather['temp']}°C | 💧 Humidity: {weather['humidity']}%")
        else:
            st.warning("⚠️ City not found → Enter manually")

    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 50)
    K = st.slider("Potassium (K)", 0, 140, 50)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

    if weather:
        temperature = weather["temp"]
        humidity = weather["humidity"]
    else:
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PREDICTION SECTION
# -------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if st.button("🚀 Predict Crop"):
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]

        st.success(f"🌾 Best Crop: {prediction[0]}")

        st.write("### 🏆 Top 3 Crops")
        top3_idx = proba.argsort()[-3:][::-1]
        crops = model.classes_

        for i in top3_idx:
            st.write(f"👉 {crops[i]} ({proba[i]*100:.2f}%)")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PERFORMANCE
# -------------------------
st.markdown("### 📊 Model Accuracy")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# -------------------------
# VISUALIZATION
# -------------------------
st.markdown("### 📈 Crop Distribution")
st.bar_chart(data['label'].value_counts())
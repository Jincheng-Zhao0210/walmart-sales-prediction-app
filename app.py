import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import base64
import matplotlib.pyplot as plt
import os
import gdown
from openai import OpenAI

# ============================================================
# 1) DOWNLOAD MODEL FROM GOOGLE DRIVE
# ============================================================

MODEL_URL = "https://drive.google.com/uc?id=15nG5Po9RjDAFa0g8wfZFlWIwuuvhdzjQ"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.write("🔄 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ============================================================
# 2) LOAD MODEL SAFELY (cloudpickle)
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        import pickle
        return pickle.load(f)


model = load_model()

# ============================================================
# 3) LOAD BACKGROUND + LOGO
# ============================================================

def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

bg64 = get_base64("background.jpg")
logo64 = get_base64("logo.png")

# ============================================================
# 4) OPENAI CLIENT (Key stored in Streamlit Secrets)
# ============================================================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_insight(title, explanation, values):
    prompt = f"""
    Explain the chart in simple business English.
    Do NOT use machine learning terms.

    Title: {title}
    Explanation: {explanation}
    Values: {values}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "(AI Insight Unavailable)"

# ============================================================
# 5) STYLING
# ============================================================

if bg64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-card {{
            background: rgba(0,0,0,0.75);
            padding: 30px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        }}
        .pred-box {{
            background: #0EA5E9;
            padding: 18px;
            border-radius: 12px;
            color: white;
            font-size: 26px;
            text-align: center;
            margin-top: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 6) HEADER
# ============================================================

st.markdown("<h1 style='text-align:center; color:#38BDF8;'>Walmart Weekly Sales Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7dd3fc;'>Forecasting • Feature Importance • AI Insights</p>", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ============================================================
# 7) USER INPUT FORM
# ============================================================

st.header("🔧 Enter Store Information")

store = st.number_input("Store ID", 1, 50, 1)
dept = st.number_input("Dept ID", 1, 99, 1)
holiday = st.selectbox("Holiday Flag", [0, 1])
temp = st.number_input("Temperature (°F)", value=70.0)
fuel = st.number_input("Fuel Price ($)", value=2.50)
cpi = st.number_input("CPI", value=220.0)
unemp = st.number_input("Unemployment (%)", value=5.0)
year = st.number_input("Year", value=2023)
month = st.number_input("Month", 1, 12, 1)
week = st.number_input("Week", 1, 53, 1)

df = pd.DataFrame({
    "Store": [store],
    "Dept": [dept],
    "Holiday_Flag": [holiday],
    "Temperature": [temp],
    "Fuel_Price": [fuel],
    "CPI": [cpi],
    "Unemployment": [unemp],
    "Year": [year],
    "Month": [month],
    "Week": [week],
})

# ============================================================
# 8) PREDICTION
# ============================================================

if st.button("Predict Weekly Sales"):
    pred = float(model.predict(df)[0])
    st.markdown(f'<div class="pred-box"><b>${pred:,.2f}</b></div>', unsafe_allow_html=True)

# ============================================================
# 9) FEATURE IMPORTANCE (Perturbation)
# ============================================================

st.subheader("📌 Feature Importance")

base = float(model.predict(df)[0])
importance = {}

for col in df.columns:
    t = df.copy()
    t[col] *= 1.10
    importance[col] = abs(float(model.predict(t)[0]) - base)

importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots()
ax.barh(list(importance.keys()), list(importance.values()), color="#38BDF8")
ax.invert_yaxis()
st.pyplot(fig)

st.write(ai_insight("Feature Importance", "Drivers of weekly sales.", importance))

# ============================================================
# 10) FORECAST 10 WEEKS
# ============================================================

st.subheader("📈 10-Week Forecast")

future_weeks = np.arange(week, week + 10)
df_future = df.loc[df.index.repeat(10)].copy()
df_future["Week"] = future_weeks

preds = model.predict(df_future)

fig2, ax2 = plt.subplots()
ax2.plot(future_weeks, preds, marker="o", color="#7dd3fc")
st.pyplot(fig2)

st.write(ai_insight("10-Week Forecast", "Predicted sales trend.", {"weeks": list(future_weeks), "sales": list(preds)}))

st.markdown("</div>", unsafe_allow_html=True)

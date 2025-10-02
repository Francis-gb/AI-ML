import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 🔗 Backend API endpoint
API_URL = "https://heatstress-backend.onrender.com/predict"

st.set_page_config(page_title="Singapore Heat Stress Predictor", layout="centered")
st.title("🌡️ Singapore Heat Stress Predictor")

st.markdown("This app uses real-time NEA sensor data to forecast WBGT at 3h, 6h & 12h.")

horizons = ["Now", "3h", "6h", "12h"]
results = []
inputs = {}

if st.button("Predict WBGT"):
    for h in horizons:
        try:
            response = requests.post(API_URL, json={"horizon": h}, timeout=10)
            response.raise_for_status()
            data = response.json()
            results.append((h, data["wbgt_prediction"]))
            if h == "Now":
                inputs = data["inputs"]
        except Exception as e:
            st.error(f"Error fetching prediction for {h}: {e}")
            results.append((h, None))

    # 🧾 Display live inputs
    st.subheader("📡 Real-Time NEA Sensor Inputs")
    st.write(f"**Temperature**: {inputs.get('temp_c', 'N/A')} °C")
    st.write(f"**Relative Humidity**: {inputs.get('rh_percent', 'N/A')} %")
    st.write(f"**Wind Speed**: {inputs.get('wind_speed_ms', 'N/A')} m/s")

    # 📊 Display WBGT chart
    st.subheader("📈 WBGT Forecast")
    df = pd.DataFrame(results, columns=["Horizon", "WBGT"])

    def wbgt_color(wbgt):
        if wbgt is None:
            return "gray"
        elif wbgt < 29.9:
            return "white"
        elif wbgt < 30.9:
            return "green"
        elif wbgt < 31.9:
            return "yellow"
        elif wbgt < 32.9:
            return "red"
        elif wbgt < 34.9:
            return "black"
        else:
            return "#8B0000"  # dark red for Cut-Off

    colors = [wbgt_color(wbgt) for wbgt in df["WBGT"]]

    fig, ax = plt.subplots()
    bars = ax.bar(df["Horizon"], df["WBGT"], color=colors)
    ax.set_ylabel("WBGT (°C)")
    ax.set_ylim(20, 45)
    ax.set_title("Predicted WBGT by Forecast Horizon")

    for bar, wbgt in zip(bars, df["WBGT"]):
        if wbgt is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{wbgt:.1f}", ha='center', va='bottom')

    st.pyplot(fig)

    # 🛡️ Heat Stress Risk Levels
    st.subheader("🛡️ Heat Stress Risk Levels")
    for h, wbgt in results:
        if wbgt is None:
            st.write(f"**{h}**: ❌ Error fetching prediction")
        elif wbgt < 29.9:
            st.write(f"**{h}**: ⚪ White — Work-Rest (min): 60–15")
        elif wbgt < 30.9:
            st.write(f"**{h}**: 🟢 Green — Work-Rest (min): 45–15")
        elif wbgt < 31.9:
            st.write(f"**{h}**: 🟡 Yellow — Work-Rest (min): 30–15")
        elif wbgt < 32.9:
            st.write(f"**{h}**: 🔴 Red — Work-Rest (min): 30–30")
        elif wbgt < 34.9:
            st.write(f"**{h}**: ⚫ Black — Work-Rest (min): 15–30")
        else:
            st.write(f"**{h}**: ❌ **Cut-Off** — No Strenuous Training")

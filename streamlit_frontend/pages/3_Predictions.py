import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import time # For live prediction simulation
import itertools # For cycling numbers

# --- Configuration ---
# Use environment variable or default for backend URL
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_data(endpoint):
    """Fetches data from the backend API."""
    url = f"{BACKEND_URL}/{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {url}.")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Timeout Error: The request to {url} timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error fetching {endpoint}: {e.response.status_code} {e.response.reason}")
        try:
            st.error(f"Backend message: {e.response.json().get('detail', 'No details provided.')}")
        except Exception: pass
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching/processing {endpoint}: {e}")
        return None

# --- Page Config ---
st.set_page_config(page_title="Predictions - OpenADR Cloud", layout="wide")

st.title("🔮 Energy Predictions")
st.markdown("Generate and view energy demand predictions.")

# --- Check Gemini Status ---
gemini_configured = False
root_info = fetch_data("")
if root_info:
    gemini_configured = root_info.get('gemini_configured', False)

if not gemini_configured:
    st.warning("⚠️ Gemini API not configured in the backend. Predictions may not be available or accurate.")

# --- Prediction Sections in Columns ---
col_hist, col_live, col_custom = st.columns(3)

# --- Column 1: Historical Prediction ---
with col_hist:
    st.subheader("Next Hour Prediction")
    st.markdown("Predicts demand for the next hour based on recent historical data.")
    if st.button("Predict Next Hour", disabled=not gemini_configured):
        with st.spinner("Requesting next hour prediction..."):
            api_url = f"{BACKEND_URL}/predict/next-hour"
            try:
                response = requests.get(api_url, timeout=20) # Longer timeout for prediction
                response.raise_for_status()
                prediction_result = response.json()
                predicted_load = prediction_result.get("predicted_total_load_next_hour")
                st.success(f"Predicted Load: **{predicted_load:.2f}**")
            except requests.exceptions.RequestException as e:
                 st.error(f"Error making prediction request: {e}")
                 try: st.error(f"Backend message: {e.response.json().get('detail', 'No details provided.')}")
                 except Exception: pass

# --- Column 2: Live Prediction (Simulated) ---
with col_live:
    st.subheader("Live Prediction (Simulated)")
    st.markdown("Simulates prediction based on live readings (cycles 1-5).")

    # Initialize session state for cycling number
    if 'live_prediction_cycle' not in st.session_state:
        st.session_state.live_prediction_cycle = itertools.cycle([1, 2, 3, 4, 5])

    if st.button("Get Next Live Prediction", disabled=not gemini_configured):
        # In a real scenario, you'd call fetch_data("predict/live") here
        # For simulation:
        simulated_value = next(st.session_state.live_prediction_cycle)
        st.metric("Simulated Live Prediction", f"{simulated_value:.2f}", help="Cycling 1-5 for demo")
        # Placeholder for actual API call handling
        # live_pred_data = fetch_data("predict/live")
        # if live_pred_data:
        #     predicted_value = live_pred_data.get('predicted_next_live_reading')
        #     count = live_pred_data.get('based_on_readings_count', 0)
        #     st.metric("Predicted Next Reading (W)", f"{predicted_value:.2f}", help=f"Based on {count} readings.")
        # else:
        #     st.warning("Could not fetch live prediction from backend.")


# --- Column 3: Custom Prediction ---
with col_custom:
    st.subheader("Custom Demand Prediction")
    if gemini_configured:
        # Fetch hourly averages to populate dropdowns/defaults
        avg_data = fetch_data("averages/hourly")
        weather_descriptions_list = ["clear sky", "few clouds", "scattered clouds", "broken clouds", "shower rain", "rain", "light rain", "thunderstorm", "snow", "mist"] # Example list
        default_hour = datetime.now().hour
        default_temp = 290.0
        default_humidity = 50
        default_wind = 5.0
        default_weather = "clear sky"

        if avg_data:
            try:
                df_avg = pd.DataFrame(avg_data)
                df_avg.set_index('hour', inplace=True)
                if default_hour in df_avg.index:
                    current_hour_avg = df_avg.loc[default_hour]
                    default_temp = current_hour_avg.get('temp', default_temp)
                    default_humidity = int(current_hour_avg.get('humidity', default_humidity))
                    default_wind = current_hour_avg.get('wind_speed', default_wind)
            except Exception as e:
                st.warning(f"Could not load defaults from averages: {e}")

        with st.form("custom_prediction_form"):
            st.write("Enter conditions:")
            hour = st.slider("Hour", 0, 23, default_hour)
            temp = st.number_input("Temp (K)", value=default_temp, format="%.2f")
            humidity = st.slider("Humidity (%)", 0, 100, default_humidity)
            wind_speed = st.number_input("Wind (m/s)", value=default_wind, min_value=0.0, format="%.2f")
            weather_description = st.selectbox("Weather", options=weather_descriptions_list, index=weather_descriptions_list.index(default_weather) if default_weather in weather_descriptions_list else 0)

            submitted = st.form_submit_button("Predict Custom Demand")

            if submitted:
                payload = {
                    "hour": hour,
                    "temp": temp,
                    "humidity": float(humidity),
                    "wind_speed": wind_speed,
                    "weather_description": weather_description
                }
                api_url = f"{BACKEND_URL}/predict/custom"
                try:
                    with st.spinner("Requesting custom prediction..."):
                        response = requests.post(api_url, json=payload, timeout=20) # Longer timeout
                        response.raise_for_status()
                        prediction_result = response.json()
                        predicted_load = prediction_result.get("predicted_total_load_custom")
                        st.success(f"Predicted Load: **{predicted_load:.2f}**")
                        with st.expander("See Input Conditions"):
                            st.json(prediction_result.get("input_conditions"))

                except requests.exceptions.RequestException as e:
                     st.error(f"Error making custom prediction request: {e}")
                     try: st.error(f"Backend message: {e.response.json().get('detail', 'No details provided.')}")
                     except Exception: pass
    else:
        st.info("Custom prediction disabled as Gemini API is not configured.")


st.markdown("---")
st.info(f"Backend API URL: {BACKEND_URL}")

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables (if needed for frontend, e.g., backend URL)
load_dotenv("../.env") # Load from the root .env file

# --- Configuration ---
# Assuming the backend runs on localhost:8000 as configured in .env
BACKEND_URL = f"http://{os.getenv('BACKEND_HOST', 'localhost')}:{os.getenv('BACKEND_PORT', '8000')}"

# --- Helper Functions ---
def get_backend_data(endpoint):
    """Fetches data from a backend API endpoint."""
    try:
        response = requests.get(f"{BACKEND_URL}/{endpoint}")
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="OpenADR Simulation Dashboard", layout="wide")

st.title("OpenADR Demand Response Simulation")

# --- Sidebar (Optional Controls) ---
st.sidebar.header("Simulation Controls")
# TODO: Add controls like triggering predictions, selecting date ranges, etc.
# Example:
# if st.sidebar.button("Predict Demand"):
#     prediction_data = get_backend_data("predict") # Assuming a /predict endpoint
#     if prediction_data:
#         st.sidebar.success("Prediction successful!")
#         # Store prediction data in session state if needed
#         st.session_state['prediction'] = prediction_data

# --- Main Dashboard Area ---
tab1, tab2, tab3 = st.tabs(["Live Status", "Historical Data", "Predictions"])

with tab1:
    st.header("Live Simulation Status")
    st.write("Displays current OpenADR events, VEN status, etc.")
    # TODO: Fetch and display live status from backend (e.g., /status endpoint)
    live_status = get_backend_data("status") # Placeholder endpoint
    if live_status:
        st.json(live_status)
    else:
        st.warning("Could not fetch live status.")

with tab2:
    st.header("Historical Energy Data (Spain)")
    st.write("Visualize consumption, generation, pricing, and weather data.")
    # TODO: Fetch historical data from backend (e.g., /historical_data endpoint)
    # Example: Fetching data and plotting
    # historical_data = get_backend_data("historical_data?timespan=...")
    # if historical_data:
    #     df = pd.DataFrame(historical_data)
    #     # Ensure 'timestamp' column is datetime type if present
    #     if 'timestamp' in df.columns:
    #         try:
    #             df['timestamp'] = pd.to_datetime(df['timestamp'])
    #             # Example plot (adjust columns as needed)
    #             fig = px.line(df, x='timestamp', y=['consumption', 'generation'], title="Energy Consumption vs Generation")
    #             st.plotly_chart(fig, use_container_width=True)
    #         except Exception as e:
    #             st.error(f"Error processing historical data: {e}")
    #             st.dataframe(df) # Show raw data if plotting fails
    #     else:
    #         st.dataframe(df) # Show raw data if no timestamp
    # else:
    #     st.warning("Could not fetch historical data.")
    st.info("Placeholder for historical data visualization.")


with tab3:
    st.header("Demand Predictions (Gemini API)")
    st.write("Displays demand predictions based on historical data and weather.")
    # TODO: Fetch prediction data from backend (e.g., /predictions endpoint)
    # prediction_data = st.session_state.get('prediction', None) # Get from session state if triggered
    # if not prediction_data:
    #     prediction_data = get_backend_data("predictions") # Or fetch latest automatically

    # if prediction_data:
    #     # Visualize prediction data (e.g., line chart, table)
    #     st.write("Predicted Demand:")
    #     # Example: Assuming prediction_data is a list of {'timestamp': ..., 'predicted_demand': ...}
    #     # pred_df = pd.DataFrame(prediction_data)
    #     # if 'timestamp' in pred_df.columns:
    #     #     pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    #     #     fig_pred = px.line(pred_df, x='timestamp', y='predicted_demand', title="Predicted Energy Demand")
    #     #     st.plotly_chart(fig_pred, use_container_width=True)
    #     # else:
    #     #     st.dataframe(pred_df)
    #     st.json(prediction_data) # Display raw prediction for now
    # else:
    #     st.warning("Could not fetch prediction data.")
    st.info("Placeholder for demand prediction visualization.")

# --- Footer ---
st.markdown("---")
st.caption("OpenADR Cloud Simulation v0.1")

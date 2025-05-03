import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="OpenADR Cloud Simulation",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help', # Replace with actual link if available
        'Report a bug': "https://www.example.com/bug", # Replace with actual link if available
        'About': "# OpenADR Cloud Simulation Dashboard\nBuilt with Streamlit."
    }
)

# --- Main Page Content ---
st.title("Welcome to the OpenADR Cloud Simulation Dashboard! ☁️")

st.markdown(
    """
    This application provides a dashboard and tools to interact with the OpenADR Cloud Simulation backend.

    **Navigate using the sidebar on the left** to explore different sections:

    - **📊 Dashboard:** View system overview, historical energy data, and average profiles.
    - **🔌 OpenADR:** (Under Construction) Monitor VEN/VTN interactions and events.
    - **🔮 Predictions:** Generate custom, historical, and live energy demand predictions.

    ---
    """
)

# Display Backend URL (optional, for info)
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
st.sidebar.info(f"Backend API: {BACKEND_URL}")

# Add any other global elements or information needed on the main landing page.

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import time

# Backend API URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="OpenADR Info - OpenADR Cloud", layout="wide")

st.title("🔌 OpenADR Information")
st.markdown("Details about VEN/VTN interactions and events are displayed here.")

# Function to fetch active events
def fetch_active_events():
    try:
        response = requests.get(f"{BACKEND_URL}/openadr/events/active")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching active events: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return []

# Function to fetch all events
def fetch_all_events(limit=50):
    try:
        response = requests.get(f"{BACKEND_URL}/openadr/events?limit={limit}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching events: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return []

# Function to create a new event
def create_event(event_data):
    try:
        response = requests.post(f"{BACKEND_URL}/openadr/events", json=event_data)
        if response.status_code == 200:
            st.success("Event created successfully!")
            return response.json()
        else:
            st.error(f"Error creating event: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None

# Function to update event status
def update_event_status(event_id, status):
    try:
        response = requests.put(f"{BACKEND_URL}/openadr/events/{event_id}/status?status={status}")
        if response.status_code == 200:
            st.success(f"Event status updated to {status}!")
            return True
        else:
            st.error(f"Error updating event status: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return False

# Function to format datetime for display
def format_datetime(dt_str):
    if isinstance(dt_str, str):
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    else:
        dt = dt_str
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

# VEN Status Section
st.subheader("VEN Status")
ven_status_col1, ven_status_col2 = st.columns(2)
with ven_status_col1:
    st.metric("VEN Client Status", "Connected", "Simulated")
with ven_status_col2:
    st.metric("Last Communication", datetime.now(timezone.utc).strftime("%H:%M:%S"), "Just now")

# Active Events Section
st.subheader("Active Events")
active_events = fetch_active_events()

if active_events:
    # Create a DataFrame for better display
    active_events_data = []
    for event in active_events:
        active_events_data.append({
            "Event ID": event["event_id"],
            "Signal Type": event["signal_type"],
            "Signal Level": event["signal_level"],
            "Start Time": format_datetime(event["start_time"]),
            "End Time": format_datetime(event["end_time"]),
            "Duration (min)": event["duration_minutes"],
            "Status": event["status"]
        })
    
    active_df = pd.DataFrame(active_events_data)
    st.dataframe(active_df, use_container_width=True)
    
    # Add action buttons for each active event
    for event in active_events:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Event: {event['event_id']}")
        with col2:
            if st.button("Mark Complete", key=f"complete_{event['event_id']}"):
                if update_event_status(event["event_id"], "completed"):
                    st.rerun()
else:
    st.info("No active events at this time.")

# Event Creation Form
st.subheader("Create New Event")
with st.expander("Create a new OpenADR event"):
    with st.form("new_event_form"):
        event_id = st.text_input("Event ID", value=f"event_{int(time.time())}")
        signal_type = st.selectbox("Signal Type", ["level", "price", "simple", "delta"])
        signal_level = st.slider("Signal Level", 0.0, 5.0, 1.0, 0.1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now(timezone.utc).date())
        with col2:
            start_time_str = st.text_input("Start Time (HH:MM)", value=datetime.now(timezone.utc).strftime("%H:%M"))
        with col3:
            duration_minutes = st.number_input("Duration (minutes)", min_value=1, value=30)
        
        target_ven_id = st.text_input("Target VEN ID", value="VEN_123")
        
        submit_button = st.form_submit_button("Create Event")
        
        if submit_button:
            try:
                # Parse the date and time inputs
                hour, minute = map(int, start_time_str.split(':'))
                start_time = datetime.combine(start_date, datetime.min.time().replace(hour=hour, minute=minute))
                
                # Make sure it's timezone-aware (UTC)
                start_time = start_time.replace(tzinfo=timezone.utc)
                
                event_data = {
                    "event_id": event_id,
                    "signal_type": signal_type,
                    "signal_level": signal_level,
                    "start_time": start_time.isoformat(),
                    "duration_minutes": duration_minutes,
                    "target_ven_id": target_ven_id,
                    "status": "active"
                }
                
                if create_event(event_data):
                    st.rerun()
            except Exception as e:
                st.error(f"Error creating event: {e}")

# Event Log Section
st.subheader("Event Log")
all_events = fetch_all_events()

if all_events:
    # Create a DataFrame for better display
    all_events_data = []
    for event in all_events:
        all_events_data.append({
            "Event ID": event["event_id"],
            "Signal Type": event["signal_type"],
            "Signal Level": event["signal_level"],
            "Start Time": format_datetime(event["start_time"]),
            "End Time": format_datetime(event["end_time"]),
            "Duration (min)": event["duration_minutes"],
            "Status": event["status"],
            "Created At": format_datetime(event["created_at"])
        })
    
    all_df = pd.DataFrame(all_events_data)
    st.dataframe(all_df, use_container_width=True)
else:
    st.info("No events in the log.")

# Power Readings Log Section
st.subheader("Power Readings Log")
with st.expander("Live Power Reading Updates", expanded=True):
    # Function to fetch the latest power reading
    def fetch_latest_power_reading():
        try:
            response = requests.get(f"{BACKEND_URL}/readings/latest")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            return None
    
    # Get the latest reading
    latest_reading = fetch_latest_power_reading()
    
    # Create a container for the power readings log
    power_log_container = st.container()
    
    # Display the latest reading if available
    if latest_reading:
        with power_log_container:
            # Initialize or get the power updates log from session state
            if 'power_updates' not in st.session_state:
                st.session_state.power_updates = []
            
            # Check if this is a new reading by comparing with the last one in our log
            is_new_reading = True
            if st.session_state.power_updates:
                last_reading = st.session_state.power_updates[0]
                # Compare timestamp and power value to determine if it's a new reading
                if (last_reading.get('timestamp') == latest_reading['timestamp'] and 
                    last_reading.get('power_watts') == latest_reading['power_watts']):
                    is_new_reading = False
            
            # Add the new reading to the log if it's different
            if is_new_reading:
                # Add a simplified entry to the log
                log_entry = {
                    'timestamp': latest_reading['timestamp'],
                    'power_watts': latest_reading['power_watts'],
                    'device_id': latest_reading['device_id'],
                    'display_time': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.power_updates.insert(0, log_entry)
                # Keep only the last 20 updates to avoid clutter
                if len(st.session_state.power_updates) > 20:
                    st.session_state.power_updates = st.session_state.power_updates[:20]
            
            # Display the log of power updates with more details
            for i, update in enumerate(st.session_state.power_updates):
                st.write(f"{update['display_time']} - {update['power_watts']:.2f} W from device {update['device_id']}")
    else:
        with power_log_container:
            st.info("No power readings available yet.")

# Add auto-refresh option
st.sidebar.subheader("Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)  # Default to enabled
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)  # Default to 10 seconds

if auto_refresh:
    st.sidebar.write(f"Page will refresh every {refresh_interval} seconds")
    # Add a script to refresh the page
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )

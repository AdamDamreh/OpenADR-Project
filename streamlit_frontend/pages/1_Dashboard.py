import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import os

# --- Configuration ---
# Use environment variable or default for backend URL
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# --- Helper Functions ---
# Using st.cache_data for caching API calls
@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_data(endpoint):
    """Fetches data from the backend API."""
    url = f"{BACKEND_URL}/{endpoint}"
    try:
        response = requests.get(url, timeout=10) # Add timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {url}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Timeout Error: The request to {url} timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error fetching data from {endpoint}: {e.response.status_code} {e.response.reason}")
        try:
            # Try to show backend error message if available
            st.error(f"Backend message: {e.response.json().get('detail', 'No details provided.')}")
        except Exception:
            pass # Ignore errors during error reporting
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {endpoint}: {e}")
        return None
    except Exception as e: # Catch unexpected errors during JSON parsing etc.
        st.error(f"An unexpected error occurred while fetching or processing data from {endpoint}: {e}")
        return None

def display_status_cards():
    """Displays status cards for devices, events, and system status."""
    st.subheader("System Overview")
    cols = st.columns(1) # Changed to 1 column as we only keep one card

    # Card 1: Latest Power Reading
    with cols[0]:
        latest_reading = fetch_data("readings/latest")
        if latest_reading and isinstance(latest_reading, dict):
            power = latest_reading.get('power_watts', 'N/A')
            timestamp_str = latest_reading.get('timestamp', 'N/A')
            # Attempt to parse and format timestamp
            try:
                # Assuming timestamp is ISO format UTC (e.g., 2023-10-27T10:30:00Z)
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Convert to local timezone for display (optional)
                ts_local = ts.astimezone()
                ts_display = ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')
            except (ValueError, TypeError):
                 ts_display = timestamp_str # Show raw string if parsing fails

            st.metric("Latest Power (W)", f"{power:.2f}" if isinstance(power, (int, float)) else power, delta=None, help=f"Timestamp: {ts_display}")
        else:
            st.metric("Latest Power (W)", "N/A", help="Could not fetch latest reading.")

# Removed Predicted Next Reading and Gemini Status cards

def display_energy_chart():
    """Displays the historical energy consumption chart."""
    st.subheader("Historical Energy Consumption (Last 48 Hours)")
    # Fetch historical data (endpoint expects 'hours' query param, defaults to 48)
    energy_data = fetch_data("data/historical?hours=48")

    if energy_data and isinstance(energy_data, list) and len(energy_data) > 0:
        try:
            df = pd.DataFrame(energy_data)
            # Ensure timestamp is datetime and handle potential errors
            df['time'] = pd.to_datetime(df['time'], errors='coerce') # Coerce errors to NaT
            df.dropna(subset=['time'], inplace=True) # Remove rows where time conversion failed
            df = df.sort_values('time')

            # Ensure 'total load actual' is numeric
            df['total load actual'] = pd.to_numeric(df['total load actual'], errors='coerce')

            if not df.empty:
                # Determine units based on typical values (simple heuristic)
                median_load = df['total load actual'].median()
                unit = "kW" if median_load > 1000 else "W" # Basic check, adjust threshold if needed
                if unit == "kW":
                    df['total load actual'] = df['total load actual'] / 1000 # Convert to kW if needed

                fig = px.line(df, x='time', y='total load actual', title="Energy Usage Over Time", labels={'time': 'Timestamp', 'total load actual': f'Total Load ({unit})'})
                fig.update_layout(xaxis_title="Time", yaxis_title=f"Power ({unit})")
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No valid historical energy data points found after processing.")

        except KeyError as e:
            st.warning(f"Historical energy data format incorrect. Missing key: {e}")
            st.dataframe(energy_data) # Show raw data if processing fails
        except Exception as e:
            st.warning(f"Could not process or plot historical energy data: {e}")
            st.dataframe(energy_data) # Show raw data if plotting fails
    elif isinstance(energy_data, list) and len(energy_data) == 0:
         st.info("No historical energy data available for the selected period.")
    else:
        # Error message already shown by fetch_data
        st.info("Could not retrieve historical energy data.")


def display_hourly_averages():
    """Displays individual charts for selected hourly average metrics."""
    st.subheader("Average Hourly Profiles (Based on Historical Data)")
    avg_data = fetch_data("averages/hourly")

    if not avg_data or not isinstance(avg_data, list) or len(avg_data) == 0:
        st.info("No hourly average data available to display.")
        # Error message already shown by fetch_data if it failed
        return # Exit the function if no data

    try:
        df_avg = pd.DataFrame(avg_data)
        if 'hour' not in df_avg.columns:
            st.warning("Hourly average data is missing the 'hour' column.")
            st.dataframe(df_avg)
            return
        df_avg = df_avg.sort_values('hour') # Ensure sorted by hour

        # Define metrics to plot individually and combined
        # Removed 'generation wind' and 'generation hard coal' as they were not found
        individual_metrics = ['generation nuclear', 'generation solar', 'generation fossil gas']
        combined_metrics = {
            "Price": ['price day ahead', 'price actual'],
            "Load": ['forecast solar', 'total load actual'] # Assuming 'load forecast' is 'forecast solar' based on typical datasets
        }

        # Create columns for layout - Changed to 2 columns per row for individual graphs
        cols1 = st.columns(2)
        cols2 = st.columns(2) # Only need 2 rows of 2 for 3 individual graphs + 1 empty slot
        cols3 = st.columns(2) # For combined graphs

        col_map = cols1 + cols2 # Flatten list for easier indexing (now 4 slots)

        # Plot individual metrics
        plot_count = 0
        for metric in individual_metrics:
            if metric in df_avg.columns:
                # Use modulo 4 for the 2x2 layout
                with col_map[plot_count % 4]: # Cycle through the 4 available slots
                    fig = px.line(df_avg, x='hour', y=metric, title=f"Avg Hourly {metric.replace('_', ' ').title()}", markers=True)
                    # Rotate x-axis labels to vertical
                    fig.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Average Value",
                        xaxis=dict(tickmode='linear', dtick=2), # Show ticks every 2 hours
                        xaxis_tickangle=0 # Rotate labels vertically
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    plot_count += 1
            else:
                st.warning(f"Metric '{metric}' not found in average data.")

        # Plot combined metrics
        plot_count = 0
        for title, metrics in combined_metrics.items():
            available_metrics = [m for m in metrics if m in df_avg.columns]
            if len(available_metrics) > 0:
                 with cols3[plot_count % len(cols3)]:
                    df_melt = df_avg[['hour'] + available_metrics].melt(id_vars=['hour'], var_name='Metric', value_name='Value')
                    fig = px.line(df_melt, x='hour', y='Value', color='Metric', title=f"Avg Hourly {title}", markers=True)
                    # Rotate x-axis labels to vertical
                    fig.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Value",
                        xaxis=dict(tickmode='linear', dtick=2),
                        xaxis_tickangle=0 # Rotate labels vertically
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    plot_count += 1
            else:
                 st.warning(f"None of the metrics for '{title}' ({', '.join(metrics)}) found in average data.")


    except Exception as e:
        st.error(f"Could not process or plot hourly average data: {e}", icon="🚨")
        st.dataframe(avg_data) # Show raw data if plotting fails

    # Removed the old combined chart and data table expander
    # Removed dangling elif/else blocks causing SyntaxError

# --- Page Layout ---
st.set_page_config(page_title="Dashboard - OpenADR Cloud", layout="wide") # Set page config here too

st.title("📊 Dashboard")
st.markdown("Overview of the system status and energy data.")

# --- Row 1: Status Cards ---
display_status_cards()

# --- Row 2: Historical Chart (Full Width) ---
display_energy_chart()

# --- Row 3: Average Hourly Profiles (Full Width) ---
display_hourly_averages()

# Add more dashboard elements here if needed below the average charts

st.markdown("---")
st.info(f"Backend API URL: {BACKEND_URL}")

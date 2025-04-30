import asyncio
import logging
# import asyncio # Commented out as VEN client task is disabled
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator # Import Pydantic
# from openleadr import OpenADRClient, enable_default_logging # Commented out OpenADR
from dotenv import load_dotenv
import os
from datetime import datetime # Added for timestamping
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

# Import services
from backend.services.data_service import load_data
from backend.services.database_service import (
    init_db,
    save_power_reading,
    get_all_live_power_readings # Import the function to get all live readings
)
# Import specific functions needed, add new ones later
from backend.services.prediction_service import (
    predict_demand_with_gemini,
    predict_custom_demand_with_gemini,
    predict_live_demand_with_gemini, # Import the new live prediction function
    genai_configured
)

# Load environment variables (e.g., for API keys, database credentials)
load_dotenv()

# Configure logging
# enable_default_logging() # Commented out OpenADR logging setup
# logger = logging.getLogger('openleadr') # Use standard logger or configure differently if needed
logging.basicConfig(level=logging.INFO) # Basic logging config
logger = logging.getLogger(__name__)

# --- Configuration ---
# VTN_URL = os.getenv("VTN_URL", "http://localhost:8080/OpenADR2/Simple/2.0b") # Commented out OpenADR config
# VEN_NAME = os.getenv("VEN_NAME", "my_ven") # Commented out OpenADR config
# VTN_ID = os.getenv("VTN_ID", "my_vtn") # Commented out OpenADR config

# --- Global variables ---
# In a production scenario, consider a more robust state management or caching solution
historical_data: pd.DataFrame | None = None
hourly_averages: pd.DataFrame | None = None # To store pre-calculated hourly averages

# --- Lifespan Context Manager (replaces on_event startup/shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global historical_data
    logger.info("Starting up backend services...")

    # Load data using the data service
    logger.info("Loading historical energy and weather data...")
    historical_data = load_data()
    global hourly_averages # Allow modification of global
    if historical_data is None:
        logger.error("Failed to load historical data. Prediction endpoints might not work.")
    else:
        logger.info("Historical data loaded successfully.")
        # Set timezone for easier slicing if not already set (assuming UTC from load_data)
        if historical_data['time'].dt.tz is None:
             logger.warning("Localizing historical data time column to UTC.")
             historical_data['time'] = historical_data['time'].dt.tz_localize('UTC')

        # --- Pre-calculate hourly averages ---
        try:
            logger.info("Calculating hourly averages from historical data...")
            # Ensure 'time' is the index for easier grouping
            if historical_data.index.name != 'time':
                historical_data_indexed = historical_data.set_index('time')
            else:
                historical_data_indexed = historical_data

            # Select numeric columns suitable for averaging (exclude object types like weather_description)
            numeric_cols = historical_data_indexed.select_dtypes(include=['number']).columns
            # Group by hour of the day and calculate mean
            hourly_averages = historical_data_indexed[numeric_cols].groupby(historical_data_indexed.index.hour).mean()
            # Add hour column for easier lookup
            hourly_averages['hour'] = hourly_averages.index
            logger.info("Hourly averages calculated successfully.")
            logger.debug(f"Hourly Averages Head:\n{hourly_averages.head()}")
        except Exception as e:
            logger.error(f"Failed to calculate hourly averages: {e}", exc_info=True)
            hourly_averages = None # Ensure it's None if calculation fails

    # Initialize the database (create tables if they don't exist)
    try:
        await init_db()
    except Exception as e:
        # Log error but allow app to potentially continue if DB isn't strictly required for all endpoints
        logger.error(f"Database initialization failed: {e}. Endpoints requiring DB may fail.", exc_info=True)
        # Depending on requirements, you might want to raise the exception here to stop startup
        # raise

    # Start the VEN client in the background - COMMENTED OUT
    # ven_task = asyncio.create_task(start_ven_client())

    # TODO: Initialize database connection pool if needed (SQLAlchemy engine is created in database_service)
    # TODO: Start VTN server if running internally - COMMENTED OUT
    # if 'vtn_server' in locals():
    #     vtn_task = asyncio.create_task(vtn_server.run())

    logger.info("Backend services (without OpenADR) started.")
    yield # Application runs here
    logger.info("Shutting down backend services...")

    # Stop VEN client - COMMENTED OUT
    # if 'ven_client' in locals() and ven_client.is_running(): # Check if defined before using
    #     logger.info("Stopping VEN client...")
    #     await ven_client.stop()
    #     try:
    #         if 'ven_task' in locals(): # Check if task exists
    #              await asyncio.wait_for(ven_task, timeout=5.0)
    #              logger.info("VEN client stopped.")
    #         else:
    #              logger.info("VEN client stopped (task not found).")
    #     except asyncio.TimeoutError:
    #         logger.warning("VEN client did not stop gracefully within timeout.")
    #     except Exception as e:
    #          logger.error(f"Error stopping VEN client: {e}")


    # TODO: Close database connections
    # TODO: Stop VTN server if running internally - COMMENTED OUT
    # if 'vtn_server' in locals() and vtn_server.is_running():
    #     await vtn_server.stop()
    #     await vtn_task # Wait for VTN task to finish

    logger.info("Backend services stopped.")


# --- Pydantic Model for Custom Prediction Input ---
class CustomPredictionInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Target hour of the day (0-23)")
    temp: float = Field(..., description="Temperature (in Kelvin, matching dataset)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage (0-100)")
    wind_speed: float = Field(..., ge=0, description="Wind speed")
    weather_description: str = Field(..., description="Text description of weather (e.g., 'clear sky', 'light rain')")
    # Add other relevant weather features if needed, e.g., pressure, clouds_all

# --- Pydantic Model for Power Meter Reading ---
class PowerReading(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the reading (UTC if possible)")
    power_watts: float = Field(..., ge=0, description="Power consumption in Watts")
    device_id: str | None = Field(None, description="Optional identifier for the device sending the data")

    @validator('timestamp', pre=True, always=True)
    def set_default_timestamp(cls, v):
        # Ensure timestamp is timezone-aware (UTC) if not provided or naive
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
             try:
                 v = datetime.fromisoformat(v)
             except ValueError:
                 raise ValueError("Invalid timestamp format. Use ISO 8601 format.")
        if v.tzinfo is None:
            logger.warning("Received naive timestamp, assuming UTC.")
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


# --- FastAPI App ---
# Use the lifespan context manager for startup/shutdown logic
app = FastAPI(title="OpenADR Cloud Simulation Backend (Prediction Only)", lifespan=lifespan)

# --- OpenLEADR VEN (Client) Setup - COMMENTED OUT ---
# async def handle_event(event):
#     """
#     Placeholder function to handle incoming OpenADR events from the VTN.
#     """
#     logger.info(f"Received event: {event}")
#     # Example: Trigger prediction based on event signal
#     # This is a simplified example; real logic depends on event structure and needs
#     try:
#         signal = event['event_signals'][0]['intervals'][0]['signal_payload']
#         logger.info(f"Event signal payload: {signal}")
#
#         # --- Example: Use prediction if event requires action ---
#         # This logic needs refinement based on actual event types and signals
#         if historical_data is not None and not historical_data.empty:
#              # Get the latest data slice (e.g., last 6 hours before the event start time)
#              # Note: Event time needs careful handling (timezones!)
#              # For simplicity, using the latest available data for now
#              latest_data_slice = historical_data.tail(6)
#              predicted_demand = await predict_demand_with_gemini(latest_data_slice)
#              if predicted_demand is not None:
#                  logger.info(f"Demand prediction triggered by event: {predicted_demand}")
#                  # TODO: Use the prediction to decide optIn/optOut or adjust load
#              else:
#                  logger.warning("Prediction failed, cannot use it for event handling.")
#         else:
#              logger.warning("No historical data available for prediction during event handling.")
#
#     except (KeyError, IndexError) as e:
#         logger.error(f"Could not parse event signal details: {e}")
#     except Exception as e:
#         logger.error(f"Error during event handling: {e}")
#
#
#     # TODO: Implement actual opt-in/opt-out logic based on event and prediction
#     return 'optIn' # Default response for now
#
# ven_client = OpenADRClient(ven_name=VEN_NAME, vtn_url=VTN_URL)
# ven_client.add_handler('on_event', handle_event)

# --- OpenLEADR VTN (Server) Setup - COMMENTED OUT ---
# In a real-world scenario, VTN and VEN might run as separate services.
# For simulation, we can include a basic VTN server here or assume an external one.
# This example assumes an external VTN or focuses primarily on the VEN side.
# If you need a VTN within this app:
# vtn_server = OpenADRServer(vtn_id=VTN_ID)
# vtn_server.add_handler('on_create_party_registration', lambda reg_info: True) # Auto-approve registrations
# TODO: Add VTN handlers for report registration, event creation, etc.

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "OpenADR Cloud Simulation Backend Running", "gemini_configured": genai_configured}

@app.get("/predict/next-hour")
async def get_next_hour_prediction():
    """
    Triggers a demand prediction for the next hour using the latest available data.
    """
    if historical_data is None or historical_data.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded or empty.")

    if not genai_configured:
         raise HTTPException(status_code=503, detail="Gemini API not configured. Check API key.")

    try:
        # --- Simulate fetching latest data based on current time ---
        now_utc = datetime.now(timezone.utc)
        # Find data up to the *start* of the current hour
        cutoff_time = now_utc.replace(minute=0, second=0, microsecond=0)
        # Define the window for historical data (e.g., 6 hours before cutoff)
        start_time = cutoff_time - timedelta(hours=6)

        logger.info(f"Current time (UTC): {now_utc}")
        logger.info(f"Simulating prediction based on data from {start_time} up to {cutoff_time}")

        # Filter the DataFrame (ensure 'time' column is datetime and timezone-aware)
        # The check historical_data['time'].dt.tz is important
        if historical_data['time'].dt.tz is None:
             # This shouldn't happen if lifespan sets it, but as a fallback
             logger.warning("Historical data time column is not timezone-aware. Assuming UTC.")
             historical_data['time'] = historical_data['time'].dt.tz_localize('UTC')

        # Select the slice of data ending *before* the cutoff time
        # Note: Slicing with tz-aware index
        data_slice = historical_data[(historical_data['time'] >= start_time) & (historical_data['time'] < cutoff_time)]

        if data_slice.empty:
             logger.warning(f"No historical data found for the window {start_time} to {cutoff_time}. Using last 6 rows as fallback.")
             # Fallback to using the absolute latest data if the time window is outside the dataset range
             data_slice = historical_data.tail(6)
             if data_slice.empty:
                  raise HTTPException(status_code=404, detail="No historical data available for prediction.")
        elif len(data_slice) < 6:
             logger.warning(f"Found only {len(data_slice)} rows for the window {start_time} to {cutoff_time}. Using available data.")
             # Proceed with fewer rows if necessary, or use tail(6) as fallback? For now, use what's found.

        logger.info(f"Using {len(data_slice)} rows for prediction input.")

        predicted_demand = await predict_demand_with_gemini(data_slice)

        if predicted_demand is None:
            raise HTTPException(status_code=500, detail="Prediction failed. Check backend logs.")

        return {"predicted_total_load_next_hour": predicted_demand}

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

@app.post("/predict/custom")
async def get_custom_prediction(input_data: CustomPredictionInput):
    """
    Triggers a demand prediction based on user-specified hour and weather conditions,
    using historical averages for that hour as context.
    """
    if historical_data is None or historical_data.empty or hourly_averages is None:
        raise HTTPException(status_code=503, detail="Historical data or hourly averages not available.")

    if not genai_configured:
         raise HTTPException(status_code=503, detail="Gemini API not configured. Check API key.")

    try:
        # Get the pre-calculated average data for the specified hour
        target_hour = input_data.hour
        avg_data_for_hour = hourly_averages.loc[target_hour]

        if avg_data_for_hour is None or avg_data_for_hour.empty:
             raise HTTPException(status_code=404, detail=f"Could not find average historical data for hour {target_hour}.")

        # Call the custom prediction service function
        predicted_demand = await predict_custom_demand_with_gemini(
            hour=target_hour,
            custom_weather=input_data,
            historical_avg_data=avg_data_for_hour
        )

        if predicted_demand is None:
            raise HTTPException(status_code=500, detail="Custom prediction failed. Check backend logs.")

        return {
            "predicted_total_load_custom": predicted_demand,
            "input_conditions": input_data.model_dump(),
            "context_hour": target_hour
        }

    except KeyError:
         raise HTTPException(status_code=404, detail=f"Could not find average historical data for hour {input_data.hour}.")
    except Exception as e:
        logger.error(f"Error in custom prediction endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during custom prediction: {e}")

@app.get("/predict/live")
async def get_live_prediction():
    """
    Triggers a prediction based on the sequence of readings currently stored
    in the live power readings database table.
    """
    if not genai_configured:
         raise HTTPException(status_code=503, detail="Gemini API not configured. Check API key.")

    try:
        # Fetch all live readings from the database
        logger.info("Fetching all live power readings from the database...")
        live_readings = await get_all_live_power_readings()

        if not live_readings:
            raise HTTPException(status_code=404, detail="No live power readings found in the database to make a prediction.")

        logger.info(f"Found {len(live_readings)} live readings. Sending to prediction service...")

        # Call the live prediction service function
        predicted_demand = await predict_live_demand_with_gemini(live_readings)

        if predicted_demand is None:
            raise HTTPException(status_code=500, detail="Live prediction failed. Check backend logs.")

        return {
            "predicted_next_live_reading": predicted_demand,
            "based_on_readings_count": len(live_readings)
        }

    except Exception as e:
        logger.error(f"Error in live prediction endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during live prediction: {e}")


@app.post("/powermeter/reading")
async def receive_power_reading(reading: PowerReading):
    """
    Receives a power meter reading from a device (simulated by SmartThings CLI).
    Initially, just logs the data. Later, will save to database.
    """
    logger.info(f"Received power reading: {reading.model_dump_json()}")

    # Save the reading to the database using the database service
    saved_reading = await save_power_reading(reading)

    if saved_reading:
        return {"message": "Power reading received and saved successfully", "data": saved_reading} # Return saved data with ID
    else:
        # Error should have been logged in save_power_reading or get_db_session
        raise HTTPException(status_code=500, detail="Failed to save power reading to the database.")


# TODO: Add endpoints for fetching historical data slices, viewing state, etc.

# --- Application Startup - COMMENTED OUT OpenADR part ---
# async def start_ven_client():
#     """Starts the OpenADR VEN client."""
#     try:
#         # Add reports if needed
#         # ven_client.add_report(...)
#         logger.info("Starting VEN client run...")
#         await ven_client.run()
#     except Exception as e:
#         logger.error(f"Error during VEN client run: {e}", exc_info=True)
#     finally:
#         logger.info("VEN client run finished.")


# --- Main Execution (for running with uvicorn directly) ---
# Note: Uvicorn needs to be run pointing to the app instance, e.g.,
# uvicorn backend.main:app --reload --port 8000
# The lifespan manager handles startup/shutdown.
if __name__ == "__main__":
    # This block is useful for direct script execution testing but not for production deployment
    print("Running Uvicorn directly from script...")
    import uvicorn
    # Ensure logging is configured if running directly
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Uvicorn server directly...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

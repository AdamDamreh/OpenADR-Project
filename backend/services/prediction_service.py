import google.generativeai as genai
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import List
# Import the Pydantic model from main - use try-except for potential circular import issues if run directly
try:
    from backend.main import CustomPredictionInput
except ImportError:
    # Define a dummy class if running the service file directly for testing
    class CustomPredictionInput:
        hour: int
        temp: float
        humidity: float
        wind_speed: float
        weather_description: str

# Import the database model for live readings
try:
    from backend.services.database_service import LivePowerReadingDB
except ImportError:
    # Define a dummy class if running the service file directly for testing
    from datetime import datetime
    class LivePowerReadingDB:
        timestamp: datetime
        power_watts: float
        device_id: str | None = None

logger = logging.getLogger(__name__)

# Load environment variables (specifically the Gemini API key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    # Depending on the application's needs, you might raise an error or handle this differently.
    # For now, we'll allow the module to load but API calls will fail.
    genai_configured = False
else:
    try:
        genai.configure(api_key=API_KEY)
        genai_configured = True
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        genai_configured = False

# Initialize the Generative Model (adjust model name if needed)
# Ensure you have access to the specified model
MODEL_NAME = "gemini-1.5-flash" # Or another suitable model like gemini-pro
model = None
if genai_configured:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info(f"Gemini model '{MODEL_NAME}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model '{MODEL_NAME}': {e}")
        genai_configured = False


def format_data_for_prompt(df_slice: pd.DataFrame) -> str:
    """
    Formats a slice of the merged DataFrame into a string prompt for Gemini.
    Focuses on recent actual load, forecasts, and key weather features.

    Args:
        df_slice: A Pandas DataFrame containing recent historical data.

    Returns:
        A string prompt for the Gemini model.
    """
    # TODO: Refine this prompt based on experimentation and desired prediction horizon.
    # Select relevant columns and summarize recent data.
    # Example: Use the last few hours of data.
    prompt = "Predict the next hour's total electrical load based on the following recent data:\n\n"

    # Select key columns (adjust as needed)
    relevant_cols = [
        'time', 'total load actual', 'total load forecast', 'price actual',
        'temp', 'humidity', 'wind_speed', 'clouds_all', 'weather_description'
    ]
    # Filter out columns that might not exist in the slice if merge had issues
    existing_cols = [col for col in relevant_cols if col in df_slice.columns]
    data_subset = df_slice[existing_cols].tail(6) # Example: use last 6 hours

    prompt += data_subset.to_string(index=False) + "\n\n"
    prompt += "Consider the time of day, recent load trends, forecasts, price, and weather conditions. Provide only the predicted numerical value for the next hour's total load (e.g., 25000.5)."

    logger.debug(f"Generated prompt for Gemini (historical):\n{prompt}")
    return prompt

def format_custom_data_for_prompt(hour: int, custom_weather: CustomPredictionInput, historical_avg_data: pd.Series) -> str:
    """
    Formats historical averages and custom weather input into a prompt for Gemini.

    Args:
        hour: The target hour (0-23).
        custom_weather: Pydantic model containing user-specified weather conditions.
        historical_avg_data: Pandas Series containing the average values for the target hour.

    Returns:
        A string prompt for the Gemini model.
    """
    prompt = f"Predict the total electrical load for hour {hour} under the following specific weather conditions, considering the typical historical averages for this hour:\n\n"
    prompt += "**Specified Conditions:**\n"
    prompt += f"- Temperature: {custom_weather.temp:.2f} K\n"
    prompt += f"- Humidity: {custom_weather.humidity:.1f} %\n"
    prompt += f"- Wind Speed: {custom_weather.wind_speed:.1f} m/s\n"
    prompt += f"- Weather: {custom_weather.weather_description}\n\n"

    prompt += "**Historical Averages for Hour {hour}:**\n"
    # Select and format key average values (adjust as needed)
    avg_cols = {
        'total load actual': 'Average Load',
        'total load forecast': 'Average Forecasted Load',
        'price actual': 'Average Price',
        'temp': 'Average Temp',
        'humidity': 'Average Humidity',
        'wind_speed': 'Average Wind Speed',
        'clouds_all': 'Average Cloud Cover'
    }
    for col, name in avg_cols.items():
        if col in historical_avg_data:
            prompt += f"- {name}: {historical_avg_data[col]:.2f}\n"

    prompt += "\nBased on the specified conditions and the historical context for this hour, provide only the predicted numerical value for the total load (e.g., 25000.5)."

    logger.debug(f"Generated prompt for Gemini (custom):\n{prompt}")
    return prompt

def format_live_data_for_prompt(live_readings: List[LivePowerReadingDB]) -> str:
    """
    Formats a list of live power readings into a string prompt for Gemini.

    Args:
        live_readings: A list of LivePowerReadingDB objects.

    Returns:
        A string prompt for the Gemini model.
    """
    prompt = "Predict the next power reading (in watts) based on the following sequence of recent live power readings:\n\n"
    prompt += "Timestamp (UTC)         | Power (Watts) | Device ID\n"
    prompt += "-------------------------|---------------|----------\n"

    # Format the last N readings (e.g., last 10 or all if fewer)
    num_readings_to_show = min(len(live_readings), 10)
    for reading in live_readings[-num_readings_to_show:]:
        ts_str = reading.timestamp.strftime('%Y-%m-%d %H:%M:%S') if reading.timestamp else "N/A"
        device_id_str = reading.device_id if reading.device_id else "N/A"
        prompt += f"{ts_str:<25}| {reading.power_watts:<13.2f} | {device_id_str}\n"

    prompt += "\nConsider the trend and recent values. Provide only the predicted numerical value for the next power reading (e.g., 550.75)."

    logger.debug(f"Generated prompt for Gemini (live):\n{prompt}")
    return prompt


async def predict_demand_with_gemini(historical_data_slice: pd.DataFrame) -> float | None:
    """
    Uses the Gemini API to predict the next hour's electrical demand.

    Args:
        historical_data_slice: A Pandas DataFrame slice containing recent
                                 historical energy and weather data.

    Returns:
        The predicted demand as a float, or None if prediction fails.
    """
    if not genai_configured or model is None:
        logger.error("Gemini API is not configured or model initialization failed. Cannot predict.")
        return None

    if historical_data_slice is None or historical_data_slice.empty:
        logger.warning("Received empty data slice for prediction.")
        return None

    prompt = format_data_for_prompt(historical_data_slice)

    try:
        logger.info(f"Sending request to Gemini model '{MODEL_NAME}'...")
        response = await model.generate_content_async(prompt) # Use async version
        logger.info("Received response from Gemini.")
        logger.debug(f"Gemini raw response text: {response.text}")

        # Attempt to parse the prediction from the response
        predicted_value_str = response.text.strip()
        predicted_demand = float(predicted_value_str)
        logger.info(f"Predicted demand: {predicted_demand}")
        return predicted_demand

    except ValueError:
        logger.error(f"Could not parse Gemini response into a float: '{predicted_value_str}'")
        return None
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call (historical): {e}", exc_info=True)
        # Log more details if available in the exception
        if hasattr(e, 'response'):
             logger.error(f"Gemini API Response Error Details: {e.response}")
        return None

async def predict_custom_demand_with_gemini(hour: int, custom_weather: CustomPredictionInput, historical_avg_data: pd.Series) -> float | None:
    """
    Uses the Gemini API to predict electrical demand for a specific hour and custom weather conditions.

    Args:
        hour: The target hour (0-23).
        custom_weather: Pydantic model containing user-specified weather conditions.
        historical_avg_data: Pandas Series containing the average values for the target hour.

    Returns:
        The predicted demand as a float, or None if prediction fails.
    """
    if not genai_configured or model is None:
        logger.error("Gemini API is not configured or model initialization failed. Cannot predict.")
        return None

    prompt = format_custom_data_for_prompt(hour, custom_weather, historical_avg_data)

    try:
        logger.info(f"Sending request to Gemini model '{MODEL_NAME}' for custom prediction...")
        response = await model.generate_content_async(prompt) # Use async version
        logger.info("Received response from Gemini for custom prediction.")
        logger.debug(f"Gemini raw response text: {response.text}")

        # Attempt to parse the prediction from the response
        predicted_value_str = response.text.strip()
        predicted_demand = float(predicted_value_str)
        logger.info(f"Predicted custom demand for hour {hour}: {predicted_demand}")
        return predicted_demand

    except ValueError:
        logger.error(f"Could not parse Gemini response into a float: '{predicted_value_str}'")
        return None
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call (custom): {e}", exc_info=True)
        # Log more details if available in the exception
        if hasattr(e, 'response'):
             logger.error(f"Gemini API Response Error Details: {e.response}")
        return None

async def predict_live_demand_with_gemini(live_readings: List[LivePowerReadingDB]) -> float | None:
    """
    Uses the Gemini API to predict the next power reading based on a sequence of live readings.

    Args:
        live_readings: A list of LivePowerReadingDB objects representing recent live data.

    Returns:
        The predicted power reading as a float, or None if prediction fails.
    """
    if not genai_configured or model is None:
        logger.error("Gemini API is not configured or model initialization failed. Cannot predict.")
        return None

    if not live_readings:
        logger.warning("Received empty list of live readings for prediction.")
        return None

    prompt = format_live_data_for_prompt(live_readings)

    try:
        logger.info(f"Sending request to Gemini model '{MODEL_NAME}' for live prediction...")
        response = await model.generate_content_async(prompt) # Use async version
        logger.info("Received response from Gemini for live prediction.")
        logger.debug(f"Gemini raw response text: {response.text}")

        # Attempt to parse the prediction from the response
        predicted_value_str = response.text.strip()
        predicted_demand = float(predicted_value_str)
        logger.info(f"Predicted live demand: {predicted_demand}")
        return predicted_demand

    except ValueError:
        logger.error(f"Could not parse Gemini response into a float: '{predicted_value_str}'")
        return None
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call (live): {e}", exc_info=True)
        # Log more details if available in the exception
        if hasattr(e, 'response'):
             logger.error(f"Gemini API Response Error Details: {e.response}")
        return None


# Example usage (for testing)
if __name__ == '__main__':
    import asyncio
    # Create some dummy data resembling the merged DataFrame slice
    dummy_data = {
        'time': pd.to_datetime(['2023-01-01 10:00:00+00:00', '2023-01-01 11:00:00+00:00', '2023-01-01 12:00:00+00:00']),
        'total load actual': [28000.0, 28500.0, 29000.0],
        'total load forecast': [27800.0, 28300.0, 28800.0],
        'price actual': [55.0, 56.0, 57.0],
        'temp': [280.0, 281.0, 282.0],
        'humidity': [60, 62, 65],
        'wind_speed': [5, 6, 5],
        'clouds_all': [20, 30, 40],
        'weather_description': ['few clouds', 'scattered clouds', 'broken clouds']
    }
    dummy_df_slice = pd.DataFrame(dummy_data)

    async def run_prediction():
        logging.basicConfig(level=logging.INFO)
        if not genai_configured:
            print("Gemini not configured. Ensure GEMINI_API_KEY is set in .env")
            return

        print("Attempting historical prediction with dummy data...")
        prediction_hist = await predict_demand_with_gemini(dummy_df_slice)
        if prediction_hist is not None:
            print(f"Predicted next hour demand (historical): {prediction_hist}")
        else:
            print("Historical prediction failed.")

        # --- Test Custom Prediction ---
        print("\nAttempting custom prediction...")
        # Create dummy input matching the Pydantic model structure
        custom_input = CustomPredictionInput(
            hour=14,
            temp=290.15, # ~17 degrees C
            humidity=75.0,
            wind_speed=3.0,
            weather_description="moderate rain"
        )
        # Create dummy average data for that hour
        dummy_avg_data = pd.Series({
            'total load actual': 26000.0,
            'total load forecast': 25800.0,
            'price actual': 60.0,
            'temp': 288.0,
            'humidity': 65.0,
            'wind_speed': 4.0,
            'clouds_all': 70.0
        })

        prediction_custom = await predict_custom_demand_with_gemini(
            hour=custom_input.hour,
            custom_weather=custom_input,
            historical_avg_data=dummy_avg_data
        )
        if prediction_custom is not None:
            print(f"Predicted custom demand: {prediction_custom}")
        else:
            print("Custom prediction failed.")

        # --- Test Live Prediction ---
        print("\nAttempting live prediction...")
        # Create some dummy live readings
        dummy_live_readings = [
            LivePowerReadingDB(timestamp=pd.to_datetime('2023-01-01 12:00:00+00:00'), power_watts=500.0, device_id='dev1'),
            LivePowerReadingDB(timestamp=pd.to_datetime('2023-01-01 12:01:00+00:00'), power_watts=510.5, device_id='dev1'),
            LivePowerReadingDB(timestamp=pd.to_datetime('2023-01-01 12:02:00+00:00'), power_watts=505.2, device_id='dev1'),
            LivePowerReadingDB(timestamp=pd.to_datetime('2023-01-01 12:03:00+00:00'), power_watts=515.8, device_id='dev1'),
        ]
        prediction_live = await predict_live_demand_with_gemini(dummy_live_readings)
        if prediction_live is not None:
            print(f"Predicted live demand: {prediction_live}")
        else:
            print("Live prediction failed.")


    asyncio.run(run_prediction())

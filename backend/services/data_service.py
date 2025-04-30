import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

# Define paths relative to the project root (where the script is likely run from)
# Adjust if necessary based on execution context
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ENERGY_DATA_PATH = os.path.join(PROJECT_ROOT, 'energy_dataset.csv')
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, 'weather_features.csv')

def load_data():
    """Loads energy and weather data from CSV files."""
    try:
        logger.info(f"Loading energy data from: {ENERGY_DATA_PATH}")
        energy_df = pd.read_csv(ENERGY_DATA_PATH)
        logger.info(f"Energy data loaded successfully. Shape: {energy_df.shape}")

        logger.info(f"Loading weather data from: {WEATHER_DATA_PATH}")
        weather_df = pd.read_csv(WEATHER_DATA_PATH)
        logger.info(f"Weather data loaded successfully. Shape: {weather_df.shape}")

        # --- Data Preprocessing ---
        # Convert time columns to datetime objects
        energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True)
        weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'], utc=True)

        # Rename columns for consistency before merging
        weather_df = weather_df.rename(columns={'dt_iso': 'time'})

        # TODO: Handle potential multiple city entries if necessary.
        # For now, assuming Valencia is the target or only city.
        # weather_df = weather_df[weather_df['city_name'] == 'Valencia'] # Example filter

        # Merge the datasets on the timestamp
        logger.info("Merging energy and weather data...")
        # Use outer merge to keep all time points, decide on handling NaNs later
        merged_df = pd.merge(energy_df, weather_df, on='time', how='outer')
        merged_df = merged_df.sort_values(by='time')
        logger.info(f"Data merged successfully. Shape: {merged_df.shape}")

        # TODO: Add more preprocessing steps as needed:
        # - Handle missing values (interpolation, forward fill, etc.)
        # - Feature selection/engineering
        # - Normalization/scaling if required by the model

        return merged_df

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Make sure CSV files are in the root directory.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        return None

if __name__ == '__main__':
    # For testing the data loading function directly
    logging.basicConfig(level=logging.INFO)
    df = load_data()
    if df is not None:
        print("Data loaded and merged successfully.")
        print(df.head())
        print(df.info())
    else:
        print("Failed to load data.")

import openleadr
import asyncio
import logging
import requests # Added for HTTP requests
import json # Added for JSON handling
import sys
import os
from datetime import datetime, timedelta, timezone
from openleadr.utils import generate_id # Removed create_message import
# Removed enums import entirely as it causes issues with older openleadr versions

# Configure logging first
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('openleadr')

# Add the project root to the Python path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database functions
DATABASE_AVAILABLE = False
try:
    # Check if SQLAlchemy is installed
    import sqlalchemy
    try:
        from backend.services.database_service import save_openadr_event, update_openadr_event_status
        DATABASE_AVAILABLE = True
        logger.info("Successfully imported database service. Events will be saved to database.")
    except ImportError:
        logger.warning("Could not import database service. Events will not be saved to database.")
except ImportError:
    logger.warning("SQLAlchemy is not installed. Events will not be saved to database.")

# Define the SmartApp endpoint URL
SMARTAPP_ENDPOINT = "https://d6b5-152-15-112-78.ngrok-free.app/openadr-event"

# Function to send event data to the SmartApp
def send_event_to_smartapp(event_data):
    """Sends event data via HTTP POST to the SmartApp endpoint."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(SMARTAPP_ENDPOINT, headers=headers, data=json.dumps(event_data), timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        logger.info(f"Successfully sent event to SmartApp. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending event to SmartApp: {e}")

# Handler called during the registration process to validate the VEN
async def on_create_party_registration(payload):
    """
    Inspect the registration info from a VEN and determine whether to accept it.
    Returns a tuple (registration_id, poll_interval) on success, or False on rejection.
    """
    ven_id = payload.get('ven_id')
    logger.info(f"Registration request received from VEN ID: {ven_id}")
    # In a real scenario, you'd validate the ven_id against a list of known/allowed VENs
    if ven_id:
        registration_id = generate_id()
        logger.info(f"Accepting registration for VEN ID: {ven_id}, assigning registration ID: {registration_id}")
        # Return the new registration ID and the suggested poll interval
        return registration_id, timedelta(seconds=10)
    else:
        logger.warning("Registration denied: No VEN ID provided.")
        return False # Indicate rejection

# Handler for event opt-in/out responses from the VEN
async def handle_event(ven_id, event_id, opt_type):
    """
    Handle VEN opt responses to events and update the database.
    """
    logger.info(f"Received event response from VEN {ven_id} for event {event_id}: {opt_type}")
    
    # Update the event status in the database if available
    if DATABASE_AVAILABLE:
        try:
            # If the VEN opts out, mark the event as cancelled
            if opt_type == 'optOut':
                logger.info(f"VEN {ven_id} opted out of event {event_id}, updating status to cancelled")
                success = await update_openadr_event_status(event_id, 'cancelled')
                if success:
                    logger.info(f"Successfully updated event {event_id} status to cancelled")
                else:
                    logger.warning(f"Failed to update event {event_id} status")
        except Exception as e:
            logger.error(f"Error updating event status in database: {e}", exc_info=True)
    
    # Return the opt type to acknowledge receipt
    return opt_type

async def main():
    # Create the OpenADR VTN Server
    server = openleadr.OpenADRServer(vtn_id='VTN_SMARTAPP_DEMO',
                                     http_host='0.0.0.0', # Listen on all available interfaces
                                     http_port=9000)  # Try a different port

    # Add the required handlers
    server.add_handler('on_create_party_registration', on_create_party_registration)
    # Check the correct event handler name for your openleadr version if needed later.
    # server.add_handler('on_event', handle_event) # Removed for now as 'on_event' is not a valid handler in this version

    # --- Add a sample event ---
    # This is a simplified example. Real events would likely be more dynamic.
    event_id = generate_id()
    signal_id = generate_id()
    now = datetime.now(timezone.utc)
    # This dictionary is illustrative and not directly used by add_event below
    event_payload = {
        'event_id': event_id,
        'modification_number': 0,
        # 'event_status': EventStatus.FAR, # Removed due to import error in older openleadr versions
        'created_date_time': now,
        'test_event': False,
        'market_context': 'http://marketcontext.example.com',
        'response_required': "always", # Use string literal
        'signals': [
            {
                'signal_id': signal_id,
                'signal_name': "SIMPLE", # Use string literal
                'signal_type': "level", # Use lowercase string literal
                'intervals': [
                    {
                        'interval_id': 0,
                        'dtstart': now + timedelta(minutes=1), # Start in 1 minute
                        'duration': timedelta(minutes=5),      # Duration 5 minutes
                        'signal_payload': 1.0                   # Example signal level
                    }
                ],
                'targets': [ # Target specific VENs or groups if needed
                    {'ven_id': 'VEN_123'} # Example: Target a specific VEN ID (replace if needed)
                ]
                # Could also use Targets.GROUP_NAME, Targets.RESOURCE_NAME etc.
            }
        ],
        'targets': [ # Event-level targets (can also be signal-specific)
             {'ven_id': 'VEN_123'} # Example: Target a specific VEN ID (replace if needed)
        ]
    }
    server.add_event(ven_id='VEN_123', # VEN ID this event is intended for
                     signal_name="SIMPLE", # Use string literal
                     signal_type="level", # Use lowercase string literal
                     intervals=[{'dtstart': now + timedelta(minutes=1), 'duration': timedelta(minutes=5), 'signal_payload': 1.0}],
                     callback=handle_event, # Use the existing placeholder handler
                     event_id=event_id,
                     response_required="always", # Use string literal
                     market_context='http://marketcontext.example.com',
                     targets=[{'ven_id': 'VEN_123'}] # Ensure targets are specified here too
                     )
    logger.info(f"Added sample event {event_id} targeting VEN_123")

    # --- Send the event data to the SmartApp and save to database ---
    # In a real scenario, you might send this when the event becomes active,
    # or based on some other trigger. Here, we send it shortly after adding.
    # We need to format the data similarly to how the SmartApp expects it.
    event_data = {
        'event_id': event_id,
        'signal_type': "level", # Send lowercase string value directly
        'start_time': (now + timedelta(minutes=1)).isoformat(),
        'duration_minutes': 5,
        'signal_level': 1.0,
        'target_ven_id': 'VEN_123', # Include target info
        'status': 'active'
    }
    
    # Save to database if available
    if DATABASE_AVAILABLE:
        try:
            logger.info(f"Saving event {event_id} to database...")
            db_event = await save_openadr_event(event_data)
            if db_event:
                logger.info(f"Successfully saved event {event_id} to database with ID {db_event.id}")
            else:
                logger.warning(f"Failed to save event {event_id} to database")
        except Exception as e:
            logger.error(f"Error saving event to database: {e}", exc_info=True)
    
    # Send to SmartApp
    smartapp_event_data = {
        'openadr_event_id': event_id,
        'signal_type': event_data['signal_type'],
        'start_time': event_data['start_time'],
        'duration_minutes': event_data['duration_minutes'],
        'signal_level': event_data['signal_level'],
        'target_ven_id': event_data['target_ven_id'],
        'action': 'SHED'  # Add an action for the SmartApp to interpret
    }
    send_event_to_smartapp(smartapp_event_data)
    # --- End send event ---


    logger.info("Starting OpenADR VTN Server...")
    # Run the server task
    try:
        server_task = asyncio.create_task(server.run())
        logger.debug("Server task created, waiting for it to complete...")
        # Keep the server running (or add other logic here)
        await server_task
    except Exception as e:
        logger.error(f"Error during server run: {e}", exc_info=True)
        raise  # Re-raise to be caught by the outer try/except


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Error starting VTN server: {e}", exc_info=True)

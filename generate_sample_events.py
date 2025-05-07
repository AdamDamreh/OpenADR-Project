#!/usr/bin/env python3
"""
Generate sample OpenADR events for testing the OpenADR Cloud simulation.
This script creates events with different signal types, levels, and durations.
"""

import requests
import json
from datetime import datetime, timezone, timedelta
import random
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

def create_event(event_data):
    """Create an OpenADR event via the backend API."""
    try:
        response = requests.post(f"{BACKEND_URL}/openadr/events", json=event_data)
        if response.status_code == 200:
            logger.info(f"Successfully created event: {event_data['event_id']}")
            return response.json()
        else:
            logger.error(f"Error creating event: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to backend: {e}")
        return None

def generate_random_event():
    """Generate a random OpenADR event."""
    now = datetime.now(timezone.utc)
    event_id = f"event_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Random signal type
    signal_types = ["level", "price", "simple", "delta"]
    signal_type = random.choice(signal_types)
    
    # Random signal level (0.5 to 3.0)
    signal_level = round(random.uniform(0.5, 3.0), 1)
    
    # Random start time (between now and 1 hour from now)
    minutes_offset = random.randint(1, 60)
    start_time = now + timedelta(minutes=minutes_offset)
    
    # Random duration (5 to 60 minutes)
    duration_minutes = random.randint(5, 60)
    
    # Target VEN ID
    target_ven_id = f"VEN_{random.randint(100, 999)}"
    
    event_data = {
        "event_id": event_id,
        "signal_type": signal_type,
        "signal_level": signal_level,
        "start_time": start_time.isoformat(),
        "duration_minutes": duration_minutes,
        "target_ven_id": target_ven_id,
        "status": "active"
    }
    
    return event_data

def generate_sample_events(count=5):
    """Generate a specified number of sample events."""
    logger.info(f"Generating {count} sample OpenADR events...")
    
    created_events = []
    for i in range(count):
        event_data = generate_random_event()
        logger.info(f"Creating event {i+1}/{count}: {event_data['event_id']}")
        
        event = create_event(event_data)
        if event:
            created_events.append(event)
        
        # Sleep briefly to avoid overwhelming the API
        time.sleep(0.5)
    
    logger.info(f"Successfully created {len(created_events)} events out of {count} attempted.")
    return created_events

def generate_specific_events():
    """Generate specific events for testing different scenarios."""
    logger.info("Generating specific test events...")
    
    now = datetime.now(timezone.utc)
    events = []
    
    # Event 1: Immediate load reduction (high priority)
    events.append({
        "event_id": f"urgent_reduction_{int(time.time())}",
        "signal_type": "level",
        "signal_level": 3.0,  # High level indicates urgent reduction
        "start_time": now.isoformat(),
        "duration_minutes": 30,
        "target_ven_id": "VEN_123",
        "status": "active"
    })
    
    # Event 2: Future scheduled reduction
    events.append({
        "event_id": f"scheduled_reduction_{int(time.time())}",
        "signal_type": "level",
        "signal_level": 2.0,
        "start_time": (now + timedelta(hours=1)).isoformat(),
        "duration_minutes": 60,
        "target_ven_id": "VEN_456",
        "status": "active"
    })
    
    # Event 3: Price signal
    events.append({
        "event_id": f"price_signal_{int(time.time())}",
        "signal_type": "price",
        "signal_level": 1.5,  # 1.5x normal price
        "start_time": (now + timedelta(minutes=30)).isoformat(),
        "duration_minutes": 120,
        "target_ven_id": "VEN_789",
        "status": "active"
    })
    
    created_events = []
    for event_data in events:
        logger.info(f"Creating specific event: {event_data['event_id']}")
        event = create_event(event_data)
        if event:
            created_events.append(event)
        time.sleep(0.5)
    
    logger.info(f"Successfully created {len(created_events)} specific events.")
    return created_events

if __name__ == "__main__":
    logger.info("Starting OpenADR event generator...")
    
    # Check if backend is available
    try:
        response = requests.get(f"{BACKEND_URL}")
        if response.status_code == 200:
            logger.info(f"Backend is available at {BACKEND_URL}")
        else:
            logger.warning(f"Backend returned status code {response.status_code}")
    except Exception as e:
        logger.error(f"Backend is not available at {BACKEND_URL}: {e}")
        logger.error("Please ensure the backend server is running.")
        exit(1)
    
    # Generate specific test events
    specific_events = generate_specific_events()
    
    # Generate random events
    random_events = generate_sample_events(3)
    
    logger.info("Event generation complete.")
    logger.info(f"Created {len(specific_events)} specific events and {len(random_events)} random events.")

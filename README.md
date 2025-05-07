# OpenADR Cloud Simulation

This project simulates an OpenADR (Open Automated Demand Response) cloud system with a virtual top node (VTN) and virtual end node (VEN) for demand response events. It includes energy data analysis, demand prediction using Gemini AI, and a complete OpenADR implementation for managing demand response events.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Setup Instructions](#setup-instructions)
5. [Running the System](#running-the-system)
6. [Configuration](#configuration)
7. [OpenADR Events](#openadr-events)
8. [Troubleshooting](#troubleshooting)

## Project Overview

This simulation demonstrates a complete OpenADR ecosystem where:

- A Virtual Top Node (VTN) server creates and sends demand response events
- A Virtual End Node (VEN) client receives and responds to these events
- A SmartApp acts as a bridge between the VTN and smart devices
- A backend API handles data processing, storage, and prediction
- A frontend visualizes energy data, predictions, and OpenADR events

The system uses historical energy data and weather information to predict future energy demand, which can be used to make informed decisions about when to issue demand response events.

## Architecture

The system follows a distributed architecture with the following components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  VTN Server     │◄────┤  Backend API    │◄────┤  Frontend       │
│  (OpenLEADR)    │     │  (FastAPI)      │     │  (Streamlit)    │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  SmartApp       │     │  Database       │
│  (Node.js)      │     │  (SQLite)       │
│                 │     │                 │
└────────┬────────┘     └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│                 │
│  Smart Devices  │
│  (Simulated)    │
│                 │
└─────────────────┘
```

## Components

### Backend API (FastAPI)

The backend API is built with FastAPI and provides endpoints for:
- Energy data access and analysis
- Demand prediction using Gemini AI
- OpenADR event management
- Power meter reading collection

### VTN Server (OpenLEADR)

The VTN server is built with OpenLEADR and is responsible for:
- Creating and sending OpenADR events
- Handling VEN registrations
- Processing event responses (opt-in/opt-out)

### SmartApp (Node.js)

The SmartApp is built with Node.js and acts as a bridge between:
- The VTN server and smart devices
- Smart devices and the backend API

### Frontend (Streamlit)

The frontend is built with Streamlit and provides:
- Energy data visualization
- Demand prediction visualization
- OpenADR event management interface
- Power meter reading log

## Setup Instructions

### Prerequisites

- Python 3.9+ for backend, VTN server, and frontend
- Node.js 14+ for SmartApp
- ngrok for exposing the SmartApp to the internet

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd OpenADR-Cloud-Simulation
   ```

2. **Set up the backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up the VTN server**:
   ```bash
   cd vtn_server
   pip install -r requirements.txt
   ```

4. **Set up the frontend**:
   ```bash
   cd streamlit_frontend
   pip install -r requirements.txt
   ```

5. **Set up the SmartApp**:
   ```bash
   cd openadr-smartapp
   npm install
   ```

6. **Configure environment variables**:
   Create a `.env` file in the root directory with the following variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   VTN_URL=http://localhost:9000/OpenADR2/Simple/2.0b
   VEN_NAME=my_ven
   VTN_ID=my_vtn
   ```

## Running the System

Start each component in a separate terminal window:

1. **Start the backend API**:
   ```bash
   cd backend
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the VTN server**:
   ```bash
   cd vtn_server
   python vtn_server.py
   ```

3. **Start the frontend**:
   ```bash
   cd streamlit_frontend
   streamlit run app.py
   ```

4. **Start ngrok to expose the SmartApp**:
   ```bash
   ngrok http 3002
   ```
   Note the HTTPS URL provided by ngrok (e.g., `https://xxxx-xxxx-xxxx-xxxx.ngrok-free.app`).

5. **Update the ngrok URL in the VTN server**:
   Edit `vtn_server/vtn_server.py` and update the `SMARTAPP_ENDPOINT` variable with the new ngrok URL:
   ```python
   SMARTAPP_ENDPOINT = "https://xxxx-xxxx-xxxx-xxxx.ngrok-free.app/openadr-event"
   ```

6. **Start the SmartApp**:
   ```bash
   cd openadr-smartapp
   node server.js
   ```

7. **Generate sample events** (optional):
   ```bash
   python generate_sample_events.py
   ```

## Configuration

### ngrok URL Configuration

The ngrok URL needs to be updated in the following locations whenever it changes:

1. **VTN Server**: In `vtn_server/vtn_server.py`, update the `SMARTAPP_ENDPOINT` variable:
   ```python
   SMARTAPP_ENDPOINT = "https://xxxx-xxxx-xxxx-xxxx.ngrok-free.app/openadr-event"
   ```

2. **SmartApp Configuration**: If you're using the SmartThings platform, update the webhook URL in your SmartApp settings.

### Database Configuration

The system uses SQLite by default. The database file is created automatically in the backend directory.

## OpenADR Events

### Understanding OpenADR Events

OpenADR (Open Automated Demand Response) is a standard for automating demand response in smart grids. It defines a communication protocol between utilities (or grid operators) and energy consumers.

In this simulation:
- The VTN server represents the utility or grid operator
- The SmartApp represents the energy consumer
- OpenADR events represent demand response signals

### Event Types

The system supports several types of OpenADR events:

1. **Level Events**: Signal a specific level of demand response (e.g., 1.0 = normal, 2.0 = moderate reduction, 3.0 = significant reduction)
2. **Price Events**: Signal a change in energy price
3. **Simple Events**: Basic on/off signals
4. **Delta Events**: Signal a specific change in energy consumption

### Event Flow

1. The VTN server creates an event and sends it to the SmartApp
2. The SmartApp processes the event and controls smart devices accordingly
3. The SmartApp sends a response (opt-in or opt-out) back to the VTN server
4. The VTN server updates the event status based on the response

### Generate Sample Events Script

The `generate_sample_events.py` script creates sample OpenADR events for testing purposes. It:

1. Connects to the backend API
2. Creates specific test events with different signal types, levels, and durations
3. Creates random events with varying parameters

This script is useful for:
- Testing the system's response to different types of events
- Demonstrating the event flow
- Populating the event log for UI testing

In a real-world scenario, events would be created based on grid conditions, weather forecasts, and energy demand predictions.

## Troubleshooting

### Common Issues

1. **VTN Server Can't Connect to SmartApp**:
   - Check if the ngrok URL is correct and up-to-date
   - Ensure the SmartApp is running
   - Check the ngrok console for any errors

2. **Database Errors**:
   - Ensure SQLAlchemy is installed (`pip install sqlalchemy`)
   - Check if the database file has the correct permissions

3. **Frontend Can't Connect to Backend**:
   - Ensure the backend API is running
   - Check if the backend URL in the frontend code is correct

4. **SmartApp Not Receiving Events**:
   - Check if the ngrok URL is correct
   - Ensure the SmartApp is running
   - Check the SmartApp logs for any errors

### Logs

Each component writes logs to the console. Check these logs for error messages and debugging information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

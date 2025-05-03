'use strict';

const express = require('express');
const bodyParser = require('body-parser');
const SmartApp = require('@smartthings/smartapp');
const axios = require('axios');

// SmartThings Developer Workspace Credentials
const CLIENT_ID = process.env.SMARTAPP_CLIENT_ID || '5434f7a5-c510-4caf-8e36-a120f40e11a0';
const CLIENT_SECRET = process.env.SMARTAPP_CLIENT_SECRET || 'ae4a4b82-af0c-4e89-aec8-eb5847676cc1';

// Backend URL for sending power meter readings
const BACKEND_URL = 'http://localhost:8000/powermeter/reading';

const server = express();
// PROFESSOR CHANGE #1: Use bodyParser.raw() instead of express.json()
server.use(bodyParser.raw({ type: '*/*' }));
const PORT = process.env.PORT || 3005;

/*
 * Create the SmartApp instance.
 */
const smartapp = new SmartApp()
  .clientId(CLIENT_ID)
  .clientSecret(CLIENT_SECRET)
  .enableEventLogging(2)
  .permissions(['r:devices:*', 'x:devices:*']);

console.log("Defining configuration page 'mainPage'...");

// Configuration page - Define what the user sees when installing the app
smartapp.page('mainPage', (context, page, configData) => {
    console.log("Executing 'mainPage' handler...");
    page.name('OpenADR Bridge & Power Monitor');
    page.description('Connects OpenADR events and sends power readings to a backend.');

    // Section to select power meters
    page.section('powerMeters', section => {
        section.deviceSetting('selectedPowerMeters')
            .capability('powerMeter')
            .permissions('r')
            .required(true)
            .multiple(true)
            .label('Select Power Meters')
            .description('Select the power meter devices to monitor.');
    });

    // Keep the info section if desired, or remove it
    page.section('info', section => {
        section.paragraphSetting('message').description('Configure power meters above.');
    });
    
    // PROFESSOR CHANGE #2: Add page.complete = true
    page.complete = true;
});

// PROFESSOR CHANGE #3: Add installed() handler
smartapp.installed(async (context) => {
    console.log('SmartApp installed!');
});

// Handler called when the app is installed or updated
smartapp.updated(async (context, updateData) => {
    console.log('SmartApp updating/installing...');
    // Unsubscribe from previous subscriptions (if any)
    await context.api.subscriptions.unsubscribeAll();

    // Subscribe to power events from selected devices
    if (context.config.selectedPowerMeters) {
        console.log('Subscribing to power events for:', context.config.selectedPowerMeters.map(d => d.deviceId));
        await context.api.subscriptions.subscribeToDevices(
            context.config.selectedPowerMeters,
            'powerMeter',
            'power',
            'powerMeterHandler'
        );
    } else {
        console.log('No power meters selected, skipping subscription.');
    }

    console.log('SmartApp installed/updated successfully!');
});

// Handler called when the app is uninstalled
smartapp.uninstalled(async (context, uninstallData) => {
    console.log('SmartApp uninstalled.');
});

// Handler for power meter events
smartapp.subscribedEventHandler('powerMeterHandler', async (context, event) => {
    console.log(`Power Event: Device=${event.deviceId}, Attribute=${event.attribute}, Value=${event.value}, Unit=${event.unit}`);

    // Prepare data payload for the backend
    const payload = {
        timestamp: new Date().toISOString(),
        power_watts: parseFloat(event.value),
        device_id: event.deviceId
    };

    console.log('Sending payload to backend:', payload);

    try {
        const response = await axios.post(BACKEND_URL, payload, {
             headers: { 'Content-Type': 'application/json' },
             timeout: 5000
        });
        console.log(`Backend response status: ${response.status}`);
    } catch (error) {
        console.error(`Error sending data to backend (${BACKEND_URL}):`, error.message);
        if (error.response) {
            console.error('Backend Error Response Status:', error.response.status);
            console.error('Backend Error Response Data:', error.response.data);
        } else if (error.request) {
            console.error('Backend No Response:', error.request);
        } else {
            console.error('Axios Setup Error:', error.message);
        }
    }
});

/*
 * Use express to host the SmartApp webhook endpoints.
 */
// PROFESSOR CHANGE #1: Removed express.json() middleware (using bodyParser.raw() above)

// Entry point for SmartThings lifecycle events (PING, CONFIGURATION, INSTALL, UPDATE, UNINSTALL)
server.post('/', async (req, res) => {
  try {
    await smartapp.handleHttpCallback(req, res);
  } catch (error) {
    console.error("Error handling SmartThings callback:", error);
    res.status(500).send('Error processing request');
  }
});

// Define a specific endpoint for the OpenLEADR VTN to call
server.post('/openadr-event', express.json(), async (req, res) => {
  console.log('Received POST request on /openadr-event');
  const eventData = req.body;
  console.log('Event Data:', eventData);

  try {
    console.log("Need to implement logic to find installed app context and send commands.");
    res.status(200).send('Event received');
  } catch (error) {
    console.error("Error processing OpenADR event:", error);
    res.status(500).send('Error processing event');
  }
});

// Test endpoint to manually send data to backend
server.get('/test-send-to-backend', express.json(), async (req, res) => {
    console.log("Received request on /test-send-to-backend");

    // Prepare a sample payload
    const testPayload = {
        timestamp: new Date().toISOString(),
        power_watts: Math.random() * 1000,
        device_id: "test-device-manual-trigger"
    };

    console.log('Sending TEST payload to backend:', testPayload);

    try {
        const response = await axios.post(BACKEND_URL, testPayload, {
             headers: { 'Content-Type': 'application/json' },
             timeout: 5000
        });
        console.log(`TEST: Backend response status: ${response.status}`);
        res.status(200).json({ message: "Test payload sent successfully.", backend_status: response.status, payload_sent: testPayload });
    } catch (error) {
        console.error(`TEST: Error sending data to backend (${BACKEND_URL}):`, error.message);
        let errorDetails = { message: error.message };
        if (error.response) {
            errorDetails.backend_status = error.response.status;
            errorDetails.backend_data = error.response.data;
        } else if (error.request) {
            errorDetails.backend_status = 'No Response';
        }
        res.status(500).json({ message: "Error sending test payload.", error: errorDetails, payload_sent: testPayload });
    }
});

/*
 * Start the Express server.
 */
server.listen(PORT, () => console.log(`SmartApp server listening on port ${PORT}`));

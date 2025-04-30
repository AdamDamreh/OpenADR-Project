'use strict';

const express = require('express');
const SmartApp = require('@smartthings/smartapp');
const axios = require('axios'); // --> ADDED: Import axios for HTTP requests

// --- IMPORTANT ---
// You will need to replace these placeholders later with actual values
// obtained from the SmartThings Developer Workspace when you register the app.
const CLIENT_ID = process.env.SMARTAPP_CLIENT_ID || '041b5fb9-1489-4444-be57-1435d0311a7a';
const CLIENT_SECRET = process.env.SMARTAPP_CLIENT_SECRET || '13d4dae5-ee1e-4252-860f-6a42615e2e7f';
// --- IMPORTANT ---

// --> ADDED: Define the backend URL
const BACKEND_URL = 'http://172.31.162.57:8000/powermeter/reading';

const server = express();
const PORT = process.env.PORT || 3005; // Port for the SmartApp server

/*
 * Create the SmartApp instance.
 */
const smartapp = new SmartApp()
  .clientId(CLIENT_ID)
  .clientSecret(CLIENT_SECRET)
  .enableEventLogging(2) // Log event lifecycle logs. Adjust level as needed (0=none, 1=error, 2=debug)
  .permissions(['r:devices:*', 'x:devices:*']); // Request read and execute permissions for all devices

console.log("Defining configuration page 'mainPage'...");

// Configuration page - Define what the user sees when installing the app
smartapp.page('mainPage', (context, page, configData) => {
    console.log("Executing 'mainPage' handler...");
    page.name('OpenADR Bridge & Power Monitor'); // --> MODIFIED: Updated name
    page.description('Connects OpenADR events and sends power readings to a backend.'); // --> MODIFIED: Updated description

    // --> MODIFIED: Section to select power meters
    page.section('powerMeters', section => {
        section.deviceSetting('selectedPowerMeters')
            .capability('powerMeter') // Filter for devices with power meter capability
            .permissions('r')         // Request read permission
            .required(true)           // Make selection mandatory
            .multiple(true)           // Allow selecting multiple devices
            .label('Select Power Meters')
            .description('Select the power meter devices to monitor.');
    });

    // Keep the info section if desired, or remove it
    page.section('info', section => {
        section.paragraphSetting('message').description('Configure power meters above.');
    });
});

// Handler called when the app is installed or updated
smartapp.updated(async (context, updateData) => {
    console.log('SmartApp updating/installing...');
    // Unsubscribe from previous subscriptions (if any)
    await context.api.subscriptions.unsubscribeAll();

    // --> ADDED: Subscribe to power events from selected devices
    if (context.config.selectedPowerMeters) {
        console.log('Subscribing to power events for:', context.config.selectedPowerMeters.map(d => d.deviceId));
        await context.api.subscriptions.subscribeToDevices(
            context.config.selectedPowerMeters, // Array of device config objects
            'powerMeter',                       // Capability
            'power',                            // Attribute to subscribe to
            'powerMeterHandler'                 // Name of the handler function
        );
    } else {
        console.log('No power meters selected, skipping subscription.');
    }

    console.log('SmartApp installed/updated successfully!');
});

// Handler called when the app is uninstalled
smartapp.uninstalled(async (context, uninstallData) => {
    console.log('SmartApp uninstalled.');
    // Clean up any resources if necessary (unsubscribeAll is usually sufficient)
});

// --> ADDED: Handler for power meter events
smartapp.subscribedEventHandler('powerMeterHandler', async (context, event) => {
    console.log(`Power Event: Device=${event.deviceId}, Attribute=${event.attribute}, Value=${event.value}, Unit=${event.unit}`);

    // Prepare data payload for the backend
    const payload = {
        timestamp: new Date().toISOString(), // Use current time in ISO format (UTC)
        power_watts: parseFloat(event.value), // Ensure value is a number
        device_id: event.deviceId
    };

    console.log('Sending payload to backend:', payload);

    try {
        const response = await axios.post(BACKEND_URL, payload, {
             headers: { 'Content-Type': 'application/json' },
             timeout: 5000 // Set a timeout (e.g., 5 seconds)
        });
        console.log(`Backend response status: ${response.status}`);
        // console.log('Backend response data:', response.data); // Optional: log response data
    } catch (error) {
        console.error(`Error sending data to backend (${BACKEND_URL}):`, error.message);
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('Backend Error Response Status:', error.response.status);
            console.error('Backend Error Response Data:', error.response.data);
        } else if (error.request) {
            // The request was made but no response was received
            console.error('Backend No Response:', error.request);
        } else {
            // Something happened in setting up the request that triggered an Error
            console.error('Axios Setup Error:', error.message);
        }
    }
});


// --- Placeholder for handling incoming calls from OpenLEADR VTN ---
// We will define a specific endpoint later for the VTN to call
// Example:
// .handle('openadrEvent', async (context, eventData) => {
//   console.log('Received OpenADR event data:', eventData);
//   // Process eventData (signal type, targets, intervals)
//   // Get selected devices: const devices = context.config.controlledDevices;
//   // Send commands to devices using context.api.devices.sendCommands(...)
// });
// --- Placeholder ---

/*
 * Use express to host the SmartApp webhook endpoints.
 */
server.use(express.json());

// Entry point for SmartThings lifecycle events (PING, CONFIGURATION, INSTALL, UPDATE, UNINSTALL)
server.post('/', async (req, res) => {
  try {
    await smartapp.handleHttpCallback(req, res);
  } catch (error) {
    console.error("Error handling SmartThings callback:", error);
    res.status(500).send('Error processing request');
  }
});

// --- Define a specific endpoint for the OpenLEADR VTN to call ---
// This endpoint will receive event notifications from the VTN.
server.post('/openadr-event', async (req, res) => {
  console.log('Received POST request on /openadr-event');
  const eventData = req.body;
  console.log('Event Data:', eventData);

  // TODO: Add authentication/validation for requests from the VTN

  try {
    // Find installed app instances that should handle this event
    // (This requires storing installation context, which the SDK handles partially)
    // For now, we'll just log it. We need to figure out how to get the
    // correct 'context' object for the installed app instance.
    // A simple approach might be to pass an installation ID or token from VTN.

    // --- Placeholder for triggering SmartApp logic ---
    // Example: await smartapp.handle('openadrEvent', null, eventData); // This won't work directly, needs context
    console.log("Need to implement logic to find installed app context and send commands.");
    // --- Placeholder ---

    res.status(200).send('Event received');
  } catch (error) {
    console.error("Error processing OpenADR event:", error);
    res.status(500).send('Error processing event');
  }
});
// --- End VTN endpoint ---

// --> ADDED: Test endpoint to manually send data to backend
server.get('/test-send-to-backend', async (req, res) => {
    console.log("Received request on /test-send-to-backend");

    // Prepare a sample payload
    const testPayload = {
        timestamp: new Date().toISOString(),
        power_watts: Math.random() * 1000, // Random power value for testing
        device_id: "test-device-manual-trigger"
    };

    console.log('Sending TEST payload to backend:', testPayload);

    try {
        const response = await axios.post(BACKEND_URL, testPayload, {
             headers: { 'Content-Type': 'application/json' },
             timeout: 5000 // Set a timeout (e.g., 5 seconds)
        });
        console.log(`TEST: Backend response status: ${response.status}`);
        res.status(200).json({ message: "Test payload sent successfully.", backend_status: response.status, payload_sent: testPayload });
    } catch (error) {
        console.error(`TEST: Error sending data to backend (${BACKEND_URL}):`, error.message);
        let errorDetails = { message: error.message };
        if (error.response) {
            console.error('TEST: Backend Error Response Status:', error.response.status);
            console.error('TEST: Backend Error Response Data:', error.response.data);
            errorDetails.backend_status = error.response.status;
            errorDetails.backend_data = error.response.data;
        } else if (error.request) {
            console.error('TEST: Backend No Response:', error.request);
            errorDetails.backend_status = 'No Response';
        } else {
            console.error('TEST: Axios Setup Error:', error.message);
        }
        res.status(500).json({ message: "Error sending test payload.", error: errorDetails, payload_sent: testPayload });
    }
});
// --- End Test endpoint ---


/*
 * Start the Express server.
 */
server.listen(PORT, () => console.log(`SmartApp server listening on port ${PORT}`));

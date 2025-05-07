const express = require('express')
const SmartApp = require('./smartapp')
const cache = require('./cache')
const axios = require('axios')

const server = express()
const PORT = process.env.PORT || 3002   // separate from other examples

/* Parse JSON for lifecycle and custom endpoints */
server.use(express.json({ verify: (req, res, buf) => { req.rawBody = buf } }))

/* SmartThings lifecycle callback */
server.post('/', (req, res) => {
  SmartApp.handleHttpCallback(req, res)
})

/* Endpoint for VTN → VEN messages */
server.post('/openadr-event', (req, res) => {
  console.log('[ADR] Received OpenADR event:', JSON.stringify(req.body, null, 2))
  
  // Extract event details
  const { 
    openadr_event_id, 
    signal_type, 
    signal_level, 
    start_time, 
    duration_minutes, 
    target_ven_id,
    action 
  } = req.body || {}
  
  // Validate required fields
  if (!action && !openadr_event_id) {
    return res.status(400).json({ error: 'Missing required fields (action or event_id)' })
  }

  // Determine command based on action or signal level
  let command = null
  if (action) {
    const upper = action.toString().toUpperCase()
    if (['SHED', 'OFF', 'REDUCE'].includes(upper)) {
      command = 'off'
    } else if (['NORMAL', 'ON', 'RESUME'].includes(upper)) {
      command = 'on'
    } else {
      return res.status(400).json({ error: `Unsupported action ${action}` })
    }
  } else if (signal_type === 'level') {
    // Use signal level to determine command
    // For example, level > 1.0 means reduce load (turn off)
    command = signal_level > 1.0 ? 'off' : 'on'
  } else {
    command = 'off' // Default to off for safety
  }

  console.log(`[ADR] Processing event ${openadr_event_id || 'unknown'} with signal_level=${signal_level || 'unknown'} → switch.${command}`)

  // Send commands to all registered switches
  const promises = []
  cache.forEach(({ api, switches }, appId) => {
    switches.forEach(id => {
      console.log(`[ADR] Sending ${command} to device ${id} (app ${appId})`)
      promises.push(api.devices.sendCommands(id, 'switch', command))
    })
  })

  // Also send power meter readings to backend for analysis
  cache.forEach(({ api, powerMeters }, appId) => {
    if (powerMeters && powerMeters.length > 0) {
      console.log(`[ADR] Requesting power readings from ${powerMeters.length} meters for event analysis`)
      powerMeters.forEach(id => {
        // This will trigger the subscription handler which sends data to backend
        promises.push(api.devices.getCapabilityStatus(id, 'powerMeter'))
      })
    }
  })

  Promise.allSettled(promises)
    .then(results => {
      const successful = results.filter(r => r.status === 'fulfilled').length
      const failed = results.filter(r => r.status === 'rejected').length
      
      console.log(`[ADR] Commands issued: ${successful} successful, ${failed} failed`)
      
      // Send response with event details and command results
      res.status(200).json({ 
        status: 'ok', 
        event_id: openadr_event_id,
        command: command,
        devices_affected: successful,
        failures: failed
      })
    })
    .catch(err => {
      console.error('[ADR] Error issuing commands', err)
      res.status(500).json({ error: 'Failed to send commands' })
    })
})

/* Test endpoint to manually send data to backend */
server.get('/test-send-to-backend', (req, res) => {
  console.log("Received request on /test-send-to-backend")
  const BACKEND_URL = 'http://localhost:8000/powermeter/reading'

  // Prepare a sample payload
  const testPayload = {
    timestamp: new Date().toISOString(),
    power_watts: Math.random() * 1000,
    device_id: "test-device-manual-trigger"
  }

  console.log('Sending TEST payload to backend:', testPayload)

  axios.post(BACKEND_URL, testPayload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: 5000
  })
  .then(response => {
    console.log(`TEST: Backend response status: ${response.status}`)
    res.status(200).json({ 
      message: "Test payload sent successfully.", 
      backend_status: response.status, 
      payload_sent: testPayload 
    })
  })
  .catch(error => {
    console.error(`TEST: Error sending data to backend (${BACKEND_URL}):`, error.message)
    let errorDetails = { message: error.message }
    if (error.response) {
      errorDetails.backend_status = error.response.status
      errorDetails.backend_data = error.response.data
    } else if (error.request) {
      errorDetails.backend_status = 'No Response'
    }
    res.status(500).json({ 
      message: "Error sending test payload.", 
      error: errorDetails, 
      payload_sent: testPayload 
    })
  })
})

server.listen(PORT, () => console.log(`OpenADR SmartApp server listening on port ${PORT}`))

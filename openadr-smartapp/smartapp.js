const { SmartApp } = require('@smartthings/smartapp')
const cache = require('./cache')
const axios = require('axios')
const BACKEND_URL = 'http://localhost:8000/powermeter/reading'

const CLIENT_ID = process.env.SMARTAPP_CLIENT_ID || '79b92a99-181d-4d59-be56-ca858b5c5d7e'
const CLIENT_SECRET = process.env.SMARTAPP_CLIENT_SECRET || 'db9a54a0-9813-4ec2-9666-4af5985666fe'

/**
 * OpenADR VEN SmartApp
 *  • Lets user pick power meters, controllable switches, optional temperature sensors
 *  • Subscribes to power / temperature events and logs them for debugging
 *  • Caches installed-app context + selected devices for use by server.js
 */
module.exports = new SmartApp()
  .clientId(CLIENT_ID)
  .clientSecret(CLIENT_SECRET)
  .enableEventLogging(2)     // pretty-print all lifecycle traffic (verbosity 2)
  .configureI18n()           // auto-generate i18n placeholders

  /* ────────────────────────────  CONFIG PAGE  ──────────────────────────── */
  .page('mainPage', (context, page) => {
    page.section('powerMeters', section => {
      section.deviceSetting('powerMeters')
        .capabilities(['powerMeter'])
        .permissions('r')
        .multiple(true)
        .required(true)
        .label('Select Power Meters')
        .description('Devices providing power (W) readings')
    })

    page.section('switches', section => {
      section.deviceSetting('switches')
        .capabilities(['switch'])
        .permissions('rx')
        .multiple(true)
        .required(true)
        .label('Switches to Control')
        .description('These devices will respond to OpenADR events')
    })


    page.complete = true
  })

  /* ────────────────────────────  INSTALL / UPDATE  ─────────────────────── */
  .installed(async context => {
    console.log('[DEBUG] installed() – appId', context.installedAppId)
    await setupSubscriptions(context)
  })

  .updated(async context => {
    console.log('[DEBUG] updated() – appId', context.installedAppId)
    await setupSubscriptions(context)
  })

  /* ────────────────────────────  EVENT HANDLERS  ───────────────────────── */
  .subscribedEventHandler('powerHandler', async (context, event) => {
    console.log(`[POWER] device=${event.deviceId} value=${event.value}${event.unit ?? ''}`)
    
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
  })


/* ----------------------- helper functions ----------------------- */

/**
 * Unsubscribe, resubscribe, and cache context data
 */
async function setupSubscriptions(context) {
  const appId = context.installedAppId

  // Clear previous subs
  await context.api.subscriptions.delete()

  // Subscribe to power readings
  if (context.config.powerMeters) {
    await context.api.subscriptions.subscribeToDevices(
      context.config.powerMeters,
      'powerMeter',
      'power',
      'powerHandler'
    )
    console.log(`[DEBUG] Subscribed to power for ${context.config.powerMeters.map(d => d.deviceId)}`)
  }


  // Cache for server-side OpenADR dispatch
  cache.set(appId, {
    api: context.api,
    switches: (context.config.switches || []).map(d => d.deviceId),
    powerMeters: (context.config.powerMeters || []).map(d => d.deviceId)
  })

  console.log(`[DEBUG] Cached context for ${appId}`)
}

/**
 * In-memory store of installed SmartApp contexts.
 * Key = installedAppId
 * Value = {
 *   api: SmartThings API handle,
 *   switches: string[],       // devices to control
 *   powerMeters: string[]     // power-meter devices
 * }
 *
 * NOTE: purely volatile – replace with a DB for production usage.
 */

const apps = new Map()

module.exports = {
  get: id => apps.get(id),
  set: (id, data) => apps.set(id, data),
  delete: id => apps.delete(id),
  forEach: cb => apps.forEach(cb)
}

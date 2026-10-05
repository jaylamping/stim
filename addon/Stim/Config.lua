-- Stim options: pure logic, no WoW calls, so tests can drive it headless.
-- Core.lua owns frames; this module owns defaults, validation, and slash parsing.
local _, ns = ...
ns = ns or {}

local Config = {}
ns.Config = Config

Config.QUEUE = 3

Config.DEFAULTS = {
  point = "CENTER", x = 0, y = -140,
  scale = 1, locked = true, count = 3, shown = true,
  pulse = true, desatDelay = 0.3,
}

-- ElvUI-style option descriptors for the panel builder. Core renders these;
-- tests assert the schema stays in sync with DEFAULTS.
function Config.schema()
  return {
    { key = "shown", type = "toggle", label = "Show suggestions" },
    { key = "locked", type = "toggle", label = "Lock icons (off = move them)" },
    { key = "scale", type = "range", label = "Scale", min = 0.5, max = 2, step = 0.05 },
    { key = "count", type = "range", label = "Spells shown", min = 1, max = 3, step = 1 },
    { key = "pulse", type = "toggle", label = "Pulse first icon" },
    { key = "desatDelay", type = "range", label = "Grey until (s away)", min = 0, max = 1, step = 0.05 },
  }
end

local function clamp(v, lo, hi)
  if type(v) ~= "number" then return nil end
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

-- Fill missing keys, drop nothing, clamp ranges. Returns db for chaining.
function Config.ensure(db)
  db = db or {}
  for k, v in pairs(Config.DEFAULTS) do
    if db[k] == nil then db[k] = v end
  end
  db.scale = clamp(tonumber(db.scale), 0.5, 2) or Config.DEFAULTS.scale
  db.count = math.max(1, math.min(Config.QUEUE, math.floor(tonumber(db.count) or Config.DEFAULTS.count)))
  db.desatDelay = clamp(tonumber(db.desatDelay), 0, 1)
  if db.desatDelay == nil then db.desatDelay = Config.DEFAULTS.desatDelay end
  db.locked = db.locked ~= false
  db.shown = db.shown ~= false
  db.pulse = db.pulse ~= false
  return db
end

-- Split "/stim scale 1.2" body into a normalized { cmd, arg }.
function Config.parse(msg)
  local cmd, arg = (msg or ""):match("^(%S*)%s*(.-)$")
  cmd = (cmd or ""):lower()
  arg = arg or ""
  return cmd, arg
end

-- Apply one command to db. Pure: returns the chat line Core should print.
-- `extras` carries frame callbacks Core injects (openPanel); tests omit it.
function Config.apply(db, cmd, arg, extras)
  cmd = (cmd or ""):lower()
  arg = arg or ""
  if cmd == "unlock" then
    db.locked = false
    return "Stim: drag the icons, then /stim lock"
  elseif cmd == "lock" then
    db.locked = true
    return nil
  elseif cmd == "scale" and tonumber(arg) then
    db.scale = clamp(tonumber(arg), 0.5, 2) or db.scale
    return "Stim scale " .. tostring(db.scale)
  elseif cmd == "count" and tonumber(arg) then
    db.count = math.max(1, math.min(Config.QUEUE, math.floor(tonumber(arg))))
    return "Stim shows " .. tostring(db.count)
  elseif cmd == "toggle" then
    db.shown = not db.shown
    return db.shown and "Stim shown" or "Stim hidden"
  elseif cmd == "pulse" then
    local a = arg:lower()
    if a == "on" then db.pulse = true elseif a == "off" then db.pulse = false else db.pulse = not db.pulse end
    return db.pulse and "Stim pulse on" or "Stim pulse off"
  elseif cmd == "desat" and tonumber(arg) then
    db.desatDelay = clamp(tonumber(arg), 0, 1) or db.desatDelay
    return "Stim grey delay " .. tostring(db.desatDelay)
  elseif cmd == "reset" then
    for k, v in pairs(Config.DEFAULTS) do db[k] = v end
    return "Stim reset"
  elseif cmd == "config" then
    if extras and extras.openPanel then extras.openPanel() end
    return nil
  else
    return "Stim: /stim unlock | lock | scale <n> | count <1-3> | pulse <on|off> | desat <s> | toggle | reset | config"
  end
end

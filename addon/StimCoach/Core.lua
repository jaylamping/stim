-- StimCoach: flashes the next few spells around your character. Every press is yours.
--
-- In combat, Forever hides energy, combo points, buffs, cooldown values and target health from addons.
-- StimCoach works only from what the game allows: your own casts, target changes, range checks, the
-- clock, and resource values read before the pull. It precomputes a recommendation for each energy
-- band, and the game itself shows the one that matches your real energy (UnitPowerPercent with a step
-- curve, drawn as the frame's alpha), so the addon never reads the hidden value.

local addonName, ns = ...
local data, Tracker, Policy = ns.data, ns.Tracker, ns.Policy

local QUEUE = 3            -- spells shown
local REFRESH = 0.2        -- seconds between recomputing the queues
local GATE_REFRESH = 0.05  -- seconds between gate updates
local SIZE, SMALL, GAP = 52, 38, 6

local defaults = { point = "CENTER", x = 0, y = -140, scale = 1, locked = true, count = QUEUE, shown = true }
local db

local tracker = Tracker.new(data)
local tracking = false
local disabled = false  -- not this spec's class
local lastKnown = {}  -- resource values from the last time they were readable
local spellAbility = {}
for i, a in ipairs(data.abilities) do spellAbility[a.name] = i end

-- Which ability a cast is. Names match the game's; a cast named "Feral Charge - Cat" still counts as
-- Feral Charge.
local function abilityOf(name)
  if not name then return nil end
  if spellAbility[name] then return spellAbility[name] end
  for i, a in ipairs(data.abilities) do
    if name:sub(1, #a.name + 1) == a.name .. " " then return i end
  end
  return nil
end

local function secret(v)
  return issecretvalue ~= nil and issecretvalue(v)
end

local function powerType(res)
  return Enum and Enum.PowerType and Enum.PowerType[res.power]
end

-- Resource values, readable out of combat. Target-bound points start every pull at zero.
local function readValues()
  local values = {}
  for r, res in ipairs(data.resources) do
    local v
    if res.pooled then
      local ok, value = pcall(UnitPower, "player", powerType(res))
      if ok and type(value) == "number" and not secret(value) then
        lastKnown[r] = value
        v = value
      else
        v = lastKnown[r] or res.start
      end
    else
      v = 0
    end
    values[r] = v
  end
  return values
end

local function inRange(spell)
  if not spell or not C_Spell or not C_Spell.IsSpellInRange then return true end
  local ok, result = pcall(C_Spell.IsSpellInRange, spell, "target")
  if not ok or result == nil or secret(result) then return true end
  return result == true or result == 1
end

local function ranges()
  return inRange(data.melee_spell), data.charge_spell ~= nil and inRange(data.charge_spell)
end

local function scorer(f, m)
  return Policy.scores(data.net, f, m)
end

---------------------------------------------------------------------- display

local root, bands, curves, plain
local gated = data.gated > 0 and C_CurveUtil and C_CurveUtil.CreateCurve and UnitPowerPercent and Enum and Enum.LuaCurveType

local function makeIcon(parent, size)
  local f = CreateFrame("Frame", nil, parent)
  f:SetSize(size, size)
  f.tex = f:CreateTexture(nil, "ARTWORK")
  f.tex:SetAllPoints()
  f.tex:SetTexCoord(0.08, 0.92, 0.08, 0.92)
  f.border = f:CreateTexture(nil, "OVERLAY")
  f.border:SetPoint("TOPLEFT", -2, 2)
  f.border:SetPoint("BOTTOMRIGHT", 2, -2)
  f.border:SetColorTexture(1, 0.75, 0.25, 0.9)
  f.border:SetDrawLayer("BACKGROUND")
  f:Hide()
  return f
end

local function makeStrip(parent)
  local s = CreateFrame("Frame", nil, parent)
  s:SetAllPoints(parent)
  s.icons = {}
  for k = 1, QUEUE do
    local icon = makeIcon(s, k == 1 and SIZE or SMALL)
    if k == 1 then
      icon:SetPoint("LEFT", s, "LEFT", 0, 0)
      -- the flash: the first icon pulses
      icon.pulse = icon:CreateAnimationGroup()
      local a = icon.pulse:CreateAnimation("Alpha")
      a:SetFromAlpha(1); a:SetToAlpha(0.55); a:SetDuration(0.45); a:SetSmoothing("IN_OUT")
      icon.pulse:SetLooping("BOUNCE")
    else
      icon:SetPoint("LEFT", s.icons[k - 1], "RIGHT", GAP, 0)
    end
    s.icons[k] = icon
  end
  return s
end

local function bandCurve(lo, hi)
  local c = C_CurveUtil.CreateCurve()
  c:SetType(Enum.LuaCurveType.Step)
  if lo <= 0 then c:AddPoint(0, 1) else c:AddPoint(0, 0); c:AddPoint(lo, 1) end
  if hi < 1 then c:AddPoint(hi, 0) end
  return c
end

local function buildDisplay()
  root = CreateFrame("Frame", "StimCoachFrame", UIParent)
  root:SetSize(SIZE + (QUEUE - 1) * (SMALL + GAP), SIZE)
  root:SetPoint(db.point, UIParent, db.point, db.x, db.y)
  root:SetScale(db.scale)
  root:SetMovable(true)
  root:EnableMouse(not db.locked)
  root:SetClampedToScreen(true)
  root:RegisterForDrag("LeftButton")
  root:SetScript("OnDragStart", function(f) if not db.locked then f:StartMoving() end end)
  root:SetScript("OnDragStop", function(f)
    f:StopMovingOrSizing()
    local point, _, _, x, y = f:GetPoint()
    db.point, db.x, db.y = point, x, y
  end)
  bands, curves = {}, {}
  if gated then
    local mx = data.resources[data.gated].max
    for b, lo in ipairs(data.bands) do
      local hi = data.bands[b + 1] or mx
      bands[b] = makeStrip(root)
      curves[b] = bandCurve(lo / mx, hi < mx and hi / mx or 1)
    end
  else
    plain = makeStrip(root)  -- no native gating available: one strip on the tracker's own estimate
  end
end

local function showQueue(strip, queue)
  for k, icon in ipairs(strip.icons) do
    local entry = k <= db.count and queue[k]
    if entry then
      local a = data.abilities[entry[1]]
      local tex = C_Spell and C_Spell.GetSpellTexture and C_Spell.GetSpellTexture(a.name)
      icon.tex:SetTexture(tex or 134400)
      -- desaturate the first icon while it's still a wait away
      if icon.tex.SetDesaturated then icon.tex:SetDesaturated(k == 1 and entry[2] > 0.3) end
      icon:Show()
      if icon.pulse and not icon.pulse:IsPlaying() then icon.pulse:Play() end
    else
      icon:Hide()
    end
  end
end

local function active()
  if disabled or not db.shown then return false end
  if tracking then return true end
  return UnitExists("target") and UnitCanAttack("player", "target") and not UnitIsDead("target")
end

local function refreshQueues()
  if not root then return end
  if not active() then root:Hide(); return end
  root:Show()
  local t = GetTime()
  local view = tracker
  if not tracking then  -- out of combat: what to open with, from the values you have now
    view = Tracker.new(data)
    view:start(t, readValues())
  end
  view:advance(t)
  local melee, charge = ranges()
  if gated then
    for b, lo in ipairs(data.bands) do showQueue(bands[b], view:predict(scorer, db.count, melee, charge, lo)) end
  else
    showQueue(plain, view:predict(scorer, db.count, melee, charge, nil))
  end
end

local function refreshGates()
  if not gated or not root or not root:IsShown() then return end
  for b, strip in ipairs(bands) do
    local ok, alpha = pcall(UnitPowerPercent, "player", powerType(data.resources[data.gated]), false, curves[b])
    if ok and alpha ~= nil then strip:SetAlpha(alpha) else strip:SetAlpha(b == 1 and 1 or 0) end
  end
end

---------------------------------------------------------------------- events

local frame = CreateFrame("Frame")
local sinceQueue, sinceGate = 0, 0

local function begin(t)
  tracker:start(t, readValues())
  tracking = true
end

local handlers = {}

function handlers.ADDON_LOADED(name)
  if name ~= addonName then return end
  StimCoachDB = StimCoachDB or {}
  for k, v in pairs(defaults) do if StimCoachDB[k] == nil then StimCoachDB[k] = v end end
  db = StimCoachDB
  local _, class = UnitClass("player")
  disabled = data.class_token ~= "" and class ~= data.class_token
  buildDisplay()
  readValues()
end

function handlers.PLAYER_REGEN_DISABLED()
  if not tracking then begin(GetTime()) end
  refreshQueues()
end

function handlers.PLAYER_REGEN_ENABLED()
  tracking = false
  refreshQueues()
end

function handlers.UNIT_SPELLCAST_SUCCEEDED(unit, _, spellID)
  if unit ~= "player" then return end
  local ok, name = pcall(C_Spell.GetSpellName, spellID)
  local i = ok and type(name) == "string" and abilityOf(name)
  if not i then return end
  if not tracking then begin(GetTime()) end  -- the opener can land before the combat flag
  tracker:cast(GetTime(), i)
  refreshQueues()
end

function handlers.PLAYER_TARGET_CHANGED()
  if tracking then tracker:retarget(GetTime()) end
  refreshQueues()
end
handlers.PLAYER_TARGET_DIED = handlers.PLAYER_TARGET_CHANGED

function handlers.UNIT_POWER_UPDATE(unit)
  if unit == "player" and not tracking then readValues() end
end

frame:SetScript("OnEvent", function(_, event, ...)
  local h = handlers[event]
  if h then h(...) end
end)
for event in pairs(handlers) do
  if event == "UNIT_SPELLCAST_SUCCEEDED" or event == "UNIT_POWER_UPDATE" then
    pcall(frame.RegisterUnitEvent, frame, event, "player")
  else
    pcall(frame.RegisterEvent, frame, event)
  end
end

frame:SetScript("OnUpdate", function(_, elapsed)
  sinceQueue, sinceGate = sinceQueue + elapsed, sinceGate + elapsed
  if sinceQueue >= REFRESH then sinceQueue = 0; refreshQueues() end
  if sinceGate >= GATE_REFRESH then sinceGate = 0; refreshGates() end
end)

---------------------------------------------------------------------- /stim

SLASH_STIMCOACH1 = "/stim"
SlashCmdList.STIMCOACH = function(msg)
  local cmd, arg = (msg or ""):match("^(%S*)%s*(.-)$")
  cmd = cmd:lower()
  if cmd == "unlock" then
    db.locked = false; root:EnableMouse(true); print("StimCoach: drag the icons, then /stim lock")
  elseif cmd == "lock" then
    db.locked = true; root:EnableMouse(false)
  elseif cmd == "scale" and tonumber(arg) then
    db.scale = tonumber(arg); root:SetScale(db.scale)
  elseif cmd == "count" and tonumber(arg) then
    db.count = math.max(1, math.min(QUEUE, math.floor(tonumber(arg))))
  elseif cmd == "toggle" then
    db.shown = not db.shown
  elseif cmd == "reset" then
    for k, v in pairs(defaults) do db[k] = v end
    root:ClearAllPoints(); root:SetPoint(db.point, UIParent, db.point, db.x, db.y); root:SetScale(db.scale)
  else
    print("StimCoach (" .. data.title .. "): /stim unlock | lock | scale <n> | count <1-" .. QUEUE .. "> | toggle | reset")
  end
  refreshQueues()
end

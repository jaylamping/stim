-- What Stim knows in combat, rebuilt from your own casts. This mirrors src/stim/addon/tracker.py
-- operation for operation; tests/test_addon.py replays the same events through both and compares.
-- Indices are 1-based here (Python's are 0-based); "wait" is action #abilities + 1.

local _, ns = ...
ns = ns or {}

local Tracker = {}
Tracker.__index = Tracker
ns.Tracker = Tracker

local INF = math.huge

local function copyArray(t)
  local c = {}
  for i = 1, #t do c[i] = t[i] end
  return c
end

local function copyMap(t)
  local c = {}
  for k, v in pairs(t) do c[k] = v end
  return c
end

function Tracker.new(data)
  local self = setmetatable({}, Tracker)
  self.d = data
  self.A = #data.abilities
  return self
end

-- The pull: resource values read just before combat (readable out of combat).
function Tracker:start(t, values)
  local d = self.d
  self.t0, self.now = t, t
  self.est = copyArray(values)
  self.ready, self.last = {}, {}
  for i = 1, self.A do self.ready[i] = t; self.last[i] = -INF end
  self.group_ready = {}
  for g = 1, d.groups do self.group_ready[g] = t end
  self.gcd_ready = t
  self.last_ability = 0
  self.buff_until = {}
  for _, b in ipairs(d.tracked_buffs) do self.buff_until[b] = t end
  self:retarget(t)
end

-- Target changed or died: per-target knowledge starts over.
function Tracker:retarget(t)
  local d = self.d
  self.target_since = t
  self.dot_until, self.debuff_until = {}, {}
  for _, i in ipairs(d.dot_abilities) do self.dot_until[i] = t end
  for _, i in ipairs(d.debuff_abilities) do self.debuff_until[i] = t end
  self.built = {}
  for j = 1, #d.points do self.built[j] = 0 end
end

function Tracker:costMult(r, t)
  t = t or self.now
  local m = 1.0
  for _, b in ipairs(self.d.tracked_buffs) do
    local buff = self.d.buffs[b]
    if self.buff_until[b] > t then
      for _, cm in ipairs(buff.cost_mult) do
        if cm.r == -1 or cm.r == r then m = m * cm.v end
      end
    end
  end
  return m
end

function Tracker:regenMult(r)
  local m = 1.0
  for _, b in ipairs(self.d.tracked_buffs) do
    local buff = self.d.buffs[b]
    if self.buff_until[b] > self.now then
      for _, rm in ipairs(buff.regen_mult) do
        if rm.r == r then m = m * rm.v end
      end
    end
  end
  return m
end

function Tracker:advance(t)
  local dt = t - self.now
  if dt > 0 then
    for _, r in ipairs(self.d.pooled) do
      local res = self.d.resources[r]
      self.est[r] = math.min(res.max, self.est[r] + res.regen * self:regenMult(r) * dt)
    end
    self.now = t
  end
end

local function pointIndex(d, r)
  for j, p in ipairs(d.points) do if p == r then return j end end
  return nil
end

local function isPooled(d, r)
  for _, p in ipairs(d.pooled) do if p == r then return true end end
  return false
end

-- You cast ability i (UNIT_SPELLCAST_SUCCEEDED for the player).
function Tracker:cast(t, i)
  local d = self.d
  local a = d.abilities[i]
  self:advance(t)
  for _, c in ipairs(a.cost) do
    if isPooled(d, c.r) then
      self.est[c.r] = math.max(0.0, self.est[c.r] - c.amt * self:costMult(c.r))
    end
  end
  local spent = 0
  if a.finisher > 0 then
    local j = pointIndex(d, a.finisher)
    if j then
      spent = self.built[j]
      self.built[j] = 0
    end
  end
  for _, g in ipairs(a.gain) do
    if isPooled(d, g.r) then
      self.est[g.r] = math.min(d.resources[g.r].max, self.est[g.r] + g.amt)
    else
      local j = pointIndex(d, g.r)
      self.built[j] = self.built[j] + 1
    end
  end
  for _, s in ipairs(a.set) do
    if isPooled(d, s.r) then
      self.est[s.r] = math.min(d.resources[s.r].max, s.v)
    end
  end
  if a.cooldown > 0 then
    self.ready[i] = t + a.cooldown
    if a.group > 0 then self.group_ready[a.group] = t + a.cooldown end
  end
  if a.gcd > 0 then self.gcd_ready = t + a.gcd end
  if a.buff > 0 then
    local b = d.buffs[a.buff]
    local dur = b.duration
    if #b.by_points > 0 then dur = b.by_points[math.min(spent, #b.by_points - 1) + 1] end
    self.buff_until[a.buff] = t + dur
  end
  if a.dot > 0 then self.dot_until[i] = t + a.dot end
  if a.debuff > 0 then self.debuff_until[i] = t + a.debuff end
  self.last[i] = t
  self.last_ability = i
end

-- Resource estimates projected to time t without changing the tracker; `known` is the gated
-- resource's value under the band hypothesis being evaluated.
function Tracker:estimates(t, known)
  local d = self.d
  local dt = math.max(0.0, t - self.now)
  local est = copyArray(self.est)
  for _, r in ipairs(d.pooled) do
    est[r] = math.min(d.resources[r].max, est[r] + d.resources[r].regen * self:regenMult(r) * dt)
  end
  if known ~= nil and d.gated > 0 then est[d.gated] = known end
  return est
end

function Tracker:affordable(i, est, t)
  local d = self.d
  for _, c in ipairs(d.abilities[i].cost) do
    if isPooled(d, c.r) and est[c.r] + 1e-9 < c.amt * self:costMult(c.r, t) then return false end
  end
  return true
end

-- The policy's input, as of the moment the GCD ends. Same order as Tracker.features_layout in Python.
function Tracker:features(inMelee, inCharge, known)
  local d = self.d
  local T, RECENT = d.T, d.RECENT
  local t = math.max(self.now, self.gcd_ready)
  local est = self:estimates(t, known)
  local out, n = {}, 0
  local function push(v) n = n + 1; out[n] = v end
  for i = 1, self.A do
    local a = d.abilities[i]
    local cd = math.max(0.0, self.ready[i] - t)
    if a.group > 0 then cd = math.max(cd, self.group_ready[a.group] - t) end
    push(math.log(1 + cd / T))
    push(cd <= 1e-9 and 1 or 0)
    push(math.min(RECENT, t - self.last[i]) / T)
    push(self:affordable(i, est, t) and 1 or 0)
  end
  for _, i in ipairs(d.dot_abilities) do push(math.max(0.0, self.dot_until[i] - t) / T) end
  for _, i in ipairs(d.debuff_abilities) do push(math.max(0.0, self.debuff_until[i] - t) / T) end
  for _, b in ipairs(d.tracked_buffs) do push(math.max(0.0, self.buff_until[b] - t) / T) end
  for _, r in ipairs(d.pooled) do push(est[r] / d.resources[r].max) end
  for j, r in ipairs(d.points) do
    local mx = d.resources[r].max
    push(math.min(self.built[j], 2 * mx) / mx)
  end
  push(math.min(t - self.t0, 300.0) / 100.0)
  push(math.min(RECENT, t - self.target_since) / T)
  push(inMelee and 1 or 0)
  push(inCharge and 1 or 0)
  for i = 1, self.A do push(self.last_ability == i and 1 or 0) end
  return out
end

-- Actions certainly impossible at the next decision: cooldown not ready, a finisher with nothing
-- built, out of range. Energy isn't masked. The last entry is "wait".
function Tracker:possible(inMelee, inCharge)
  local d = self.d
  local t = math.max(self.now, self.gcd_ready)
  local ok = {}
  for i = 1, self.A do
    local a = d.abilities[i]
    local yes = true
    if self.ready[i] > t + 1e-9 or (a.group > 0 and self.group_ready[a.group] > t + 1e-9) then yes = false end
    if a.finisher > 0 then
      local j = pointIndex(d, a.finisher)
      if j and self.built[j] == 0 then yes = false end
    end
    if a.gap_closer then
      yes = yes and inCharge
    elseif a.targeted and not inMelee and a.range_max == 0 then
      yes = false
    end
    ok[i] = yes
  end
  ok[self.A + 1] = true
  return ok
end

function Tracker:copy()
  local c = setmetatable({}, Tracker)
  for k, v in pairs(self) do c[k] = v end
  c.est, c.ready, c.group_ready, c.last, c.built = copyArray(self.est), copyArray(self.ready),
    copyArray(self.group_ready), copyArray(self.last), copyArray(self.built)
  c.buff_until, c.dot_until, c.debuff_until = copyMap(self.buff_until), copyMap(self.dot_until), copyMap(self.debuff_until)
  return c
end

-- The next n spells if you follow the recommendations: list of {action, seconds until due}.
-- scores(features, possible) returns the policy's scores. A "wait" advances a copy by waitStep.
function Tracker:predict(scores, n, inMelee, inCharge, known, waitStep, maxWait)
  waitStep, maxWait = waitStep or 0.5, maxWait or 3.0
  local c = self:copy()
  local start = math.max(c.now, c.gcd_ready)
  local t = start
  if known ~= nil and c.d.gated > 0 then
    c:advance(t)
    c.est[c.d.gated] = known
  end
  local out, waited = {}, 0.0
  local WAIT = self.A + 1
  while #out < n do
    local f, m = c:features(inMelee, inCharge), c:possible(inMelee, inCharge)
    local s = scores(f, m)
    local a, best = nil, -INF
    for i = 1, #m do
      if m[i] and (a == nil or s[i] > best) then a, best = i, s[i] end
    end
    if a == WAIT and waited < maxWait then
      t = t + waitStep
      waited = waited + waitStep
      c:advance(t)
      c.gcd_ready = math.max(c.gcd_ready, t)
    elseif a == WAIT then
      break
    else
      out[#out + 1] = {a, t - start}
      c:cast(t, a)
      t = math.max(c.now, c.gcd_ready)
      waited = 0.0
    end
  end
  return out
end

return Tracker

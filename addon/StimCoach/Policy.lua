-- The addon's policy: a small MLP from tracker features to a score per action (abilities, then "wait").
-- Weights come from Data.lua, exported by scripts/export_addon.py.

local _, ns = ...
ns = ns or {}

local Policy = {}
ns.Policy = Policy

local function dense(layer, x, relu)
  local w, b = layer.w, layer.b
  local out = {}
  for o = 1, #b do
    local row, s = w[o], b[o]
    for i = 1, #x do s = s + row[i] * x[i] end
    if relu and s < 0 then s = 0 end
    out[o] = s
  end
  return out
end

-- Scores for each action; impossible actions get -inf.
function Policy.scores(net, features, possible)
  local x = {}
  for i = 1, #features do x[i] = (features[i] - net.mean[i]) / net.std[i] end
  local layers = net.layers
  for k = 1, #layers - 1 do x = dense(layers[k], x, true) end
  local s = dense(layers[#layers], x, false)
  for i = 1, #s do
    if not possible[i] then s[i] = -math.huge end
  end
  return s
end

return Policy

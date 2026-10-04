"""The Lua addon must agree with the Python tracker and policy it was trained against. These tests replay
the same events through both, and load the whole addon against a mocked WoW API."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

lupa = pytest.importorskip("lupa")
from lupa import lua51  # noqa: E402

from stim.addon.export import data_lua  # noqa: E402
from stim.addon.policy import AddonNet  # noqa: E402
from stim.addon.tracker import Tracker  # noqa: E402
from stim.data import AddonView  # noqa: E402
from stim.policies import AplPolicy, EpsilonPolicy  # noqa: E402
from stim.scenarios import make_scenario  # noqa: E402
from stim.spec import load_spec  # noqa: E402

ADDON = Path(__file__).resolve().parents[1] / "addon" / "StimCoach"


class Recorder(Tracker):
    """A tracker that logs its top-level events so they can be replayed in Lua."""

    def __init__(self, spec):
        super().__init__(spec)
        self.log, self.depth = [], 0

    def _call(self, name, *args):
        if self.depth == 0:
            self.log.append((name, *args))
        self.depth += 1
        try:
            return getattr(Tracker, name)(self, *args)
        finally:
            self.depth -= 1

    def start(self, t, values):
        return self._call("start", t, list(values))

    def retarget(self, t):
        return self._call("retarget", t)

    def advance(self, t):
        return self._call("advance", t)

    def cast(self, t, ai):
        return self._call("cast", t, ai)


def small_net(tracker, seed=0):
    torch.manual_seed(seed)
    net = AddonNet(tracker.width, tracker.A + 1, 16, torch.randn(tracker.width) * 0.1, torch.rand(tracker.width) + 0.5)
    return net.double().eval()


def lua_addon(spec, net, files=("Tracker", "Policy")):
    L = lua51.LuaRuntime(unpack_returned_tuples=True)
    ns = L.table()
    load = L.execute("return function(code, ns, name) local f, err = loadstring(code, name)\n"
                     "if not f then error(err) end; return f('StimCoach', ns) end")
    load(data_lua(spec, net), ns, "Data.lua")
    for name in files:
        load((ADDON / f"{name}.lua").read_text(), ns, f"{name}.lua")
    return L, ns


def play(spec_name, kind, seed, net, checks=40):
    """Play a fight on the priority list (with some random presses), logging tracker events and, at
    decisions, what the Python side computes."""
    spec = load_spec(spec_name)
    sim = make_scenario(spec, kind, random.Random(seed))
    view = AddonView(sim)
    rec = Recorder(spec)
    rec.start(sim.t, list(sim.res))
    view.tracker = rec
    policy = EpsilonPolicy(AplPolicy(spec), 0.2, random.Random(seed))

    def scores(f, m):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(f, dtype=np.float64))[None], torch.from_numpy(np.asarray(m))[None])[0].numpy()

    n = 0
    while not sim.done and n < checks:
        opts = sim.legal_options()
        view.update()
        if len(opts) > 1:
            melee, charge = view.ranges()
            known = rec.band_of(sim.res[rec.gated])
            f = rec.features(melee, charge, known)
            m = rec.possible(melee, charge)
            rec.log.append(("check", melee, charge, known, f, m, scores(f, m), rec.predict(scores, 3, melee, charge, known)))
            n += 1
        sim.step(policy(sim, opts))
    return spec, rec.log


@pytest.mark.parametrize("kind,seed", [("boss", 1), ("trash", 2), ("boss_adds", 3)])
def test_lua_tracker_and_policy_match_python(kind, seed):
    spec = load_spec("feral")
    net = small_net(Tracker(spec))
    spec, log = play("feral", kind, seed, net)
    L, ns = lua_addon(spec, net)
    tr = ns.Tracker.new(ns.data)
    scorer = L.execute("return function(Policy, net) return function(f, m) return Policy.scores(net, f, m) end end")(
        ns.Policy, ns.data.net)
    checks = 0
    for entry in log:
        kind_ = entry[0]
        if kind_ == "start":
            tr.start(tr, entry[1], L.table_from(entry[2]))
        elif kind_ == "retarget":
            tr.retarget(tr, entry[1])
        elif kind_ == "advance":
            tr.advance(tr, entry[1])
        elif kind_ == "cast":
            tr.cast(tr, entry[1], entry[2] + 1)
        else:
            _, melee, charge, known, f, m, s, queue = entry
            lf = np.array(list(tr.features(tr, melee, charge, known).values()))
            np.testing.assert_allclose(lf, f, atol=1e-5)
            lm = np.array(list(tr.possible(tr, melee, charge).values()))
            assert (lm == m).all()
            ls = np.array(list(ns.Policy.scores(ns.data.net, L.table_from(f.astype(float).tolist()),
                                                L.table_from(m.tolist())).values()))
            finite = np.isfinite(s)
            assert (np.isfinite(ls) == finite).all()
            np.testing.assert_allclose(ls[finite], s[finite], rtol=1e-6, atol=1e-6)
            lq = [(int(e[1]) - 1, float(e[2])) for e in tr.predict(tr, scorer, 3, melee, charge, known).values()]
            assert [a for a, _ in lq] == [a for a, _ in queue]
            np.testing.assert_allclose([d for _, d in lq], [d for _, d in queue], atol=1e-6)
            checks += 1
    assert checks >= 10


WOW_MOCK = r"""
local now, energy = 0, 100
function SetMock(t, e) now = t; energy = e end
function GetTime() return now end
Enum = { PowerType = { Mana = 0, Rage = 1, Energy = 3, ComboPoints = 4 }, LuaCurveType = { Step = 1 } }
function UnitPower(unit, pt) if pt == 3 then return energy elseif pt == 0 then return 5000 end return 0 end
C_CurveUtil = { CreateCurve = function()
  local c = { pts = {} }
  function c:SetType() end
  function c:AddPoint(x, y) self.pts[#self.pts + 1] = { x, y } end
  function c:Evaluate(x) local v = 0; for _, p in ipairs(self.pts) do if x >= p[1] then v = p[2] end end; return v end
  return c
end }
function UnitPowerPercent(unit, pt, predicted, curve) return curve:Evaluate(pt == 3 and energy / 100 or 1) end
SPELLS = { [5221] = "Shred", [1082] = "Claw" }
C_Spell = { GetSpellName = function(id) return SPELLS[id] end,
            GetSpellTexture = function(name) return "tex:" .. name end,
            IsSpellInRange = function(name, unit) return true end }
function UnitExists() return true end
function UnitCanAttack() return true end
function UnitIsDead() return false end
function UnitClass() return "Druid", "DRUID" end
function issecretvalue() return false end
SlashCmdList = {}
function print() end
FRAMES = {}
local function stub() end
function CreateFrame(kind, name, parent)
  local f = { scripts = {}, events = {}, shown = true, alpha = 1, name = name }
  for _, m in ipairs({ "SetSize", "SetPoint", "ClearAllPoints", "SetScale", "SetMovable", "SetClampedToScreen",
                        "RegisterForDrag", "EnableMouse", "StartMoving", "StopMovingOrSizing", "SetAllPoints" }) do f[m] = stub end
  function f:SetScript(n, fn) self.scripts[n] = fn end
  function f:RegisterEvent(e) self.events[e] = true end
  function f:RegisterUnitEvent(e) self.events[e] = true end
  function f:Show() self.shown = true end
  function f:Hide() self.shown = false end
  function f:IsShown() return self.shown end
  function f:SetAlpha(a) self.alpha = a end
  function f:GetPoint() return "CENTER", nil, "CENTER", 0, 0 end
  function f:CreateTexture()
    local t = {}
    for _, m in ipairs({ "SetAllPoints", "SetTexCoord", "SetPoint", "SetColorTexture", "SetDrawLayer" }) do t[m] = stub end
    function t:SetTexture(v) self.texture = v end
    function t:SetDesaturated(v) self.desaturated = v end
    return t
  end
  function f:CreateAnimationGroup()
    local g = { playing = false }
    function g:CreateAnimation() local a = {}; for _, m in ipairs({ "SetFromAlpha", "SetToAlpha", "SetDuration", "SetSmoothing" }) do a[m] = stub end; return a end
    function g:SetLooping() end
    function g:Play() self.playing = true end
    function g:IsPlaying() return self.playing end
    return g
  end
  FRAMES[#FRAMES + 1] = f
  if name then _G[name] = f end
  return f
end
UIParent = CreateFrame("Frame")
"""


def test_addon_loads_and_the_game_gates_the_right_band():
    spec = load_spec("feral")
    net = small_net(Tracker(spec))
    L = lua51.LuaRuntime(unpack_returned_tuples=True)
    L.execute(WOW_MOCK)
    ns = L.table()
    load = L.execute("return function(code, ns, name) local f, err = loadstring(code, name)\n"
                     "if not f then error(err) end; return f('StimCoach', ns) end")
    load(data_lua(spec, net), ns, "Data.lua")
    for name in ("Tracker", "Policy", "Core"):
        load((ADDON / f"{name}.lua").read_text(), ns, f"{name}.lua")
    g = L.globals()
    events = next(f for f in g.FRAMES.values() if f.scripts.OnEvent is not None)
    fire = lambda *a: events.scripts.OnEvent(events, *a)  # noqa: E731
    fire("ADDON_LOADED", "StimCoach")
    g.SetMock(10.0, 100)
    fire("PLAYER_REGEN_DISABLED")
    fire("UNIT_SPELLCAST_SUCCEEDED", "player", "cast-1", 5221)  # Shred
    for energy, band in ((55.0, 3), (5.0, 1), (95.0, 5)):
        g.SetMock(11.5, energy)
        events.scripts.OnUpdate(events, 0.25)
        root = g.StimCoachFrame
        assert root.shown
        strips = [f for f in g.FRAMES.values() if f.icons is not None]
        assert len(strips) == len(spec_bands := Tracker(spec).bands())
        alphas = [s.alpha for s in strips]
        assert alphas == [1 if i == band - 1 else 0 for i in range(len(spec_bands))], alphas
        first = strips[band - 1].icons[1]
        assert first.shown and str(first.tex.texture).startswith("tex:")
    assert g.SlashCmdList.STIMCOACH is not None
    g.SlashCmdList.STIMCOACH("count 2")
    assert g.StimCoachDB.count == 2
    # a rank- or form-suffixed spell name still counts as the ability
    g.SPELLS[16979] = "Feral Charge - Cat"
    fire("UNIT_SPELLCAST_SUCCEEDED", "player", "cast-2", 16979)
    events.scripts.OnUpdate(events, 0.25)


def test_addon_stays_hidden_for_other_classes():
    spec = load_spec("feral")
    net = small_net(Tracker(spec))
    L = lua51.LuaRuntime(unpack_returned_tuples=True)
    L.execute(WOW_MOCK + '\nfunction UnitClass() return "Mage", "MAGE" end')
    ns = L.table()
    load = L.execute("return function(code, ns, name) local f, err = loadstring(code, name)\n"
                     "if not f then error(err) end; return f('StimCoach', ns) end")
    load(data_lua(spec, net), ns, "Data.lua")
    for name in ("Tracker", "Policy", "Core"):
        load((ADDON / f"{name}.lua").read_text(), ns, f"{name}.lua")
    g = L.globals()
    events = next(f for f in g.FRAMES.values() if f.scripts.OnEvent is not None)
    events.scripts.OnEvent(events, "ADDON_LOADED", "StimCoach")
    events.scripts.OnEvent(events, "PLAYER_REGEN_DISABLED")
    events.scripts.OnUpdate(events, 0.25)
    assert not g.StimCoachFrame.shown

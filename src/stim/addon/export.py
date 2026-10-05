"""Export a spec's rules and a trained addon policy as the Stim addon's Data.lua.

Everything the Lua tracker needs comes from the same compiled rules the Python tracker uses, so the two
can't disagree about a cost or a cooldown. Lua indices are 1-based; Python's are 0-based."""

from __future__ import annotations

from pathlib import Path

import torch

from ..sim import rules_for
from ..spec import Spec, load_spec
from .policy import AddonNet, load_addon
from .tracker import RECENT, T, Tracker

INTERFACE = 16001  # WoW Forever client 1.60.1
POWER = {"energy": "Energy", "rage": "Rage", "mana": "Mana"}
CLASS_TOKEN = {"Druid": "DRUID", "Rogue": "ROGUE", "Warrior": "WARRIOR", "Paladin": "PALADIN",
               "Shaman": "SHAMAN", "Hunter": "HUNTER", "Mage": "MAGE", "Priest": "PRIEST", "Warlock": "WARLOCK"}


def lua(v, indent: str = "") -> str:
    """A Python value as a Lua literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "nil"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.9g}" if v == v and abs(v) != float("inf") else ("math.huge" if v > 0 else "-math.huge")
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, dict):
        inner = indent + "  "
        items = ",\n".join(f"{inner}{k} = {lua(x, inner)}" for k, x in v.items())
        return "{\n" + items + "\n" + indent + "}" if items else "{}"
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return "{" + ", ".join(lua(float(x) if isinstance(x, float) else x) for x in v) + "}"
        inner = indent + "  "
        return "{\n" + ",\n".join(inner + lua(x, inner) for x in v) + "\n" + indent + "}" if v else "{}"
    raise TypeError(type(v))


def spec_table(spec: Spec, net: AddonNet | None = None) -> dict:
    R = rules_for(spec)
    tr = Tracker(spec)
    resources = []
    for r, res in enumerate(R.res):
        resources.append({
            "id": res.id, "kind": res.kind, "max": float(res.max), "start": float(res.start),
            "regen": float(tr.regen[r]), "pooled": r in tr.pooled,
            "power": "ComboPoints" if (res.on_target and res.kind == "secondary") else POWER.get(res.kind, ""),
        })
    buffs = []
    for bi, b in enumerate(R.buffs):
        cm = dict(R.cost_mult_buffs).get(bi, [])
        rm = dict(R.regen_mult_buffs).get(bi, [])
        buffs.append({
            "id": b.id, "duration": float(b.duration), "by_points": [float(x) for x in b.duration_by_points],
            "cost_mult": [{"r": (k + 1 if k >= 0 else -1), "v": float(v)} for k, v in cm],
            "regen_mult": [{"r": k + 1, "v": float(v)} for k, v in rm],
        })
    abilities = []
    for ai, a in enumerate(R.abilities):
        abilities.append({
            "id": a.id, "name": a.name, "key": a.key,
            "cost": [{"r": r + 1, "amt": float(amt)} for r, amt in R.cost[ai]],
            "gain": [{"r": r + 1, "amt": float(amt)} for r, amt in R.gain[ai]],
            "set": [{"r": r + 1, "v": float(v)} for r, v in R.set[ai]],
            "cooldown": float(a.cooldown), "group": R.group[ai] + 1, "gcd": float(R.gcd[ai]),
            "finisher": R.finisher[ai] + 1, "buff": R.buff_of[ai] + 1,
            "dot": float(a.dot.ticks * a.dot.interval) if a.dot else 0.0,
            "debuff": float(R.debuffs[R.debuff_of[ai]].duration) if R.debuff_of[ai] >= 0 else 0.0,
            "targeted": bool(R.targeted[ai]), "gap_closer": bool(a.gap_closer), "range_max": float(a.range_max),
        })
    melee = next((a.name for ai, a in enumerate(R.abilities)
                  if R.targeted[ai] and not a.gap_closer and not a.range_max and not a.req_behind and a.damage), None)
    charge = next((a.name for a in R.abilities if a.gap_closer), None)
    out = {
        "spec": spec.spec_name.lower(), "title": spec.name, "class_token": CLASS_TOKEN.get(spec.class_name, ""),
        "T": T, "RECENT": RECENT, "groups": R.n_groups,
        "resources": resources, "buffs": buffs, "abilities": abilities,
        "pooled": [r + 1 for r in tr.pooled], "points": [r + 1 for r in tr.points],
        "gated": tr.gated + 1, "bands": tr.bands(),
        "dot_abilities": [ai + 1 for ai in tr.dot_abilities], "debuff_abilities": [ai + 1 for ai in tr.debuff_abilities],
        "tracked_buffs": [bi + 1 for bi in tr.buffs],
        "melee_spell": melee, "charge_spell": charge, "layout": tr.features_layout(),
    }
    if net is not None:
        out["net"] = net_table(net)
    return out


def net_table(net: AddonNet) -> dict:
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    return {
        "mean": net.mean.tolist(),
        "std": net.std.tolist(),
        "layers": [{"w": layer.weight.detach().tolist(), "b": layer.bias.detach().tolist()} for layer in layers],
    }


def data_lua(spec: Spec, net: AddonNet) -> str:
    return ("-- Generated by scripts/export_addon.py from specs/ and a trained policy. Don't edit by hand.\n"
            "local _, ns = ...\nns = ns or {}\n"
            f"ns.data = {lua(spec_table(spec, net))}\nreturn ns.data\n")


def toc(spec: Spec) -> str:
    return "\n".join([
        f"## Interface: {INTERFACE}",
        "## Title: Stim",
        f"## Notes: Flashes your next spells ({spec.name}), learned by simulation. Every press is yours.",
        "## Author: stim",
        "## Version: 0.1.0",
        "## SavedVariables: StimDB",
        "",
        "Data.lua",
        "Tracker.lua",
        "Policy.lua",
        "Config.lua",
        "Core.lua",
        "",
    ])


def write_addon(spec_name: str, model_path: str | Path, out: Path) -> Path:
    spec = load_spec(spec_name)
    net, ck = load_addon(model_path)
    if ck["spec"] != spec_name:
        raise ValueError(f"{model_path} was trained for {ck['spec']}, not {spec_name}")
    if ck["layout"] != Tracker(spec).features_layout():
        raise ValueError(f"{model_path} was trained on a different feature layout; retrain it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "Data.lua").write_text(data_lua(spec, net))
    (out / "Stim.toc").write_text(toc(spec))
    return out / "Data.lua"

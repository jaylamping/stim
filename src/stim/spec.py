"""Class specs as data: resources, buffs, debuffs, procs, and abilities, loaded from TOML.

The engine implements generic melee mechanics; a spec only composes them. Nothing in the engine
knows which class it is simulating.
"""

from __future__ import annotations

import copy
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SPECS_DIR = Path(__file__).resolve().parents[2] / "specs"

RESOURCE_KINDS = ("energy", "rage", "mana", "secondary")
SCHOOLS = ("physical", "bleed", "nature", "fire", "frost", "arcane", "holy", "shadow")
PROC_TRIGGERS = ("white_hit", "yellow_hit", "hit", "crit", "white_crit", "yellow_crit", "builder_crit", "dodge")


class SpecError(ValueError):
    pass


@dataclass
class Weapon:
    damage: float
    speed: float


@dataclass
class Resource:
    id: str
    kind: str
    max: float
    start: float
    regen_per_sec: float = 0.0
    tick_amount: float = 0.0
    tick_interval: float = 0.0
    per_white_damage: float = 0.0
    on_target: bool = False  # combo points that live on the target (switching targets loses them)


@dataclass
class Aura:
    """A player buff or a target debuff."""

    id: str
    name: str
    duration: float = 0.0
    duration_by_points: list[float] = field(default_factory=list)
    max_stacks: int = 1
    charges: int = 0
    consumed_by: str = ""  # "cost", "swing", "ability:<id>", or a damage school (debuffs)
    haste: float = 0.0
    damage_mult: float = 0.0
    crit_bonus: float = 0.0
    cost_mult: dict[str, float] = field(default_factory=dict)  # resource id or "all"
    regen_mult: dict[str, float] = field(default_factory=dict)
    cleave_targets: int = 0
    cleave_radius: float = 8.0
    damage_taken: dict[str, float] = field(default_factory=dict)  # school -> bonus per stack


@dataclass
class Damage:
    weapon: float = 0.0
    flat: float = 0.0
    by_points: list[float] = field(default_factory=list)
    school: str = "physical"
    can_crit: bool = True
    crit_bonus: float = 0.0
    aoe_radius: float = 0.0  # > 0: hits enemies around the player instead of one target
    max_targets: int = 1
    extra_targets: int = 0  # cleave: also hits this many enemies near the target


@dataclass
class Dot:
    ticks: int
    interval: float
    per_tick: float = 0.0
    per_tick_per_point: float = 0.0
    school: str = "bleed"


@dataclass
class Extra:
    """Converts leftover resource into damage (Ferocious Bite, Execute)."""

    resource: str
    max: float
    damage_per: float


@dataclass
class Proc:
    id: str
    trigger: str
    chance: float = 0.0
    ppm: float = 0.0
    buff: str = ""
    debuff: str = ""
    gain: dict[str, float] = field(default_factory=dict)
    extra_attacks: int = 0
    damage: Damage | None = None
    dot: Dot | None = None
    requires_buff: str = ""


@dataclass
class Ability:
    id: str
    name: str
    key: str = ""
    cost: dict[str, float] = field(default_factory=dict)
    gain: dict[str, float] = field(default_factory=dict)
    set: dict[str, float] = field(default_factory=dict)
    cooldown: float = 0.0
    cooldown_group: str = ""
    gcd: float | None = None
    off_gcd: bool = False
    cast_time: float = 0.0
    finisher: str = ""  # resource consumed in full; damage/dot/buff scale with the amount
    damage: Damage | None = None
    dot: Dot | None = None
    extra: Extra | None = None
    buff: str = ""
    debuff: str = ""
    consumes_buff: str = ""
    next_swing: bool = False
    gap_closer: bool = False
    range_min: float = 0.0
    range_max: float = 0.0  # 0 means melee range
    req_behind: bool = False
    req_target_health_below: float = 0.0
    req_target_health_above: float = 0.0
    req_buff: str = ""
    req_no_buff: str = ""

    @property
    def aoe(self) -> bool:
        return self.damage is not None and self.damage.aoe_radius > 0


@dataclass
class Spec:
    class_name: str
    spec_name: str
    notes: str
    gcd: float
    crit_chance: float
    crit_multiplier: float
    melee_range: float
    run_speed: float
    miss_chance: float
    dodge_chance: float
    glancing_chance: float
    glancing_multiplier: float
    dual_wield_miss: float
    main_hand: Weapon
    off_hand: Weapon | None
    resources: list[Resource]
    buffs: list[Aura]
    debuffs: list[Aura]
    procs: list[Proc]
    abilities: list[Ability]
    resource_value: dict[str, float]
    reference_dps: float
    apl: list[tuple[str, str]]

    @property
    def name(self) -> str:
        return f"{self.spec_name} {self.class_name}"

    def ability_index(self, ability_id: str) -> int:
        for i, a in enumerate(self.abilities):
            if a.id == ability_id:
                return i
        raise KeyError(ability_id)

    def randomized(self, rng: random.Random, strength: float = 1.0) -> Spec:
        """A copy with perturbed tuning, so the model learns to read the numbers instead of memorizing them."""
        s = copy.deepcopy(self)

        def j(lo: float, hi: float) -> float:
            return 1.0 + (rng.uniform(lo, hi) - 1.0) * strength

        s.main_hand.damage *= j(0.75, 1.3)
        if s.off_hand:
            s.off_hand.damage *= j(0.75, 1.3)
        s.crit_chance = min(0.6, max(0.05, s.crit_chance * j(0.7, 1.35)))
        for p in s.procs:
            p.chance *= j(0.6, 1.5)
            p.ppm *= j(0.6, 1.5)
        for a in s.abilities:
            if a.damage:
                a.damage.weapon *= j(0.85, 1.15)
                a.damage.flat *= j(0.8, 1.25)
                a.damage.by_points = [d * j(0.8, 1.25) for d in a.damage.by_points]
            if a.dot:
                a.dot.per_tick *= j(0.75, 1.3)
                a.dot.per_tick_per_point *= j(0.75, 1.3)
            for r in list(a.cost):
                if self._resource(r).kind != "secondary":
                    a.cost[r] = round(a.cost[r] * j(0.85, 1.15))
        return s

    def _resource(self, rid: str) -> Resource:
        for r in self.resources:
            if r.id == rid:
                return r
        raise KeyError(rid)


# ---------------------------------------------------------------------- loading


def _take(d: dict, where: str, allowed: set[str]) -> dict:
    unknown = set(d) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return d


def _num_map(d: dict, where: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in _take(d, where, set(d)).items()}


def _damage(d: dict, where: str) -> Damage:
    _take(d, where, {"weapon", "flat", "by_points", "school", "can_crit", "crit_bonus", "aoe_radius", "max_targets", "extra_targets"})
    dmg = Damage(
        weapon=float(d.get("weapon", 0)),
        flat=float(d.get("flat", 0)),
        by_points=[float(x) for x in d.get("by_points", [])],
        school=d.get("school", "physical"),
        can_crit=bool(d.get("can_crit", True)),
        crit_bonus=float(d.get("crit_bonus", 0)),
        aoe_radius=float(d.get("aoe_radius", 0)),
        max_targets=int(d.get("max_targets", 1 if not d.get("aoe_radius") else 10)),
        extra_targets=int(d.get("extra_targets", 0)),
    )
    if dmg.school not in SCHOOLS:
        raise SpecError(f"{where}: school must be one of {SCHOOLS}")
    return dmg


def _aura(aid: str, d: dict, where: str) -> Aura:
    _take(d, where, {"name", "duration", "duration_by_points", "max_stacks", "charges", "consumed_by", "haste",
                     "damage_mult", "crit_bonus", "cost_mult", "regen_mult", "cleave_targets", "cleave_radius", "damage_taken"})
    cost_mult = d.get("cost_mult", {})
    if isinstance(cost_mult, (int, float)):
        cost_mult = {"all": cost_mult}
    return Aura(
        id=aid,
        name=d.get("name", aid.replace("_", " ").title()),
        duration=float(d.get("duration", 0)),
        duration_by_points=[float(x) for x in d.get("duration_by_points", [])],
        max_stacks=int(d.get("max_stacks", 1)),
        charges=int(d.get("charges", 0)),
        consumed_by=d.get("consumed_by", ""),
        haste=float(d.get("haste", 0)),
        damage_mult=float(d.get("damage_mult", 0)),
        crit_bonus=float(d.get("crit_bonus", 0)),
        cost_mult=_num_map(cost_mult, where + ".cost_mult"),
        regen_mult=_num_map(d.get("regen_mult", {}), where + ".regen_mult"),
        cleave_targets=int(d.get("cleave_targets", 0)),
        cleave_radius=float(d.get("cleave_radius", 8.0)),
        damage_taken=_num_map(d.get("damage_taken", {}), where + ".damage_taken"),
    )


ABILITY_FIELDS = {
    "id", "name", "key", "cost", "gain", "set", "cooldown", "cooldown_group", "gcd", "off_gcd", "cast_time",
    "finisher", "damage", "dot", "extra", "buff", "debuff", "consumes_buff", "next_swing", "gap_closer", "range",
    "requires",
}


def _dot(d: dict, where: str) -> Dot:
    _take(d, where, {"ticks", "interval", "per_tick", "per_tick_per_point", "school"})
    dot = Dot(
        ticks=int(d["ticks"]),
        interval=float(d["interval"]),
        per_tick=float(d.get("per_tick", 0)),
        per_tick_per_point=float(d.get("per_tick_per_point", 0)),
        school=d.get("school", "bleed"),
    )
    if dot.school not in SCHOOLS:
        raise SpecError(f"{where}: school must be one of {SCHOOLS}")
    return dot


def _ability(d: dict, i: int) -> Ability:
    where = f"abilities[{i}] ({d.get('id', '?')})"
    _take(d, where, ABILITY_FIELDS)
    if "id" not in d:
        raise SpecError(f"{where}: missing id")
    req = _take(d.get("requires", {}), where + ".requires", {"behind", "target_health_below", "target_health_above", "buff", "no_buff"})
    extra = d.get("extra")
    if extra is not None:
        _take(extra, where + ".extra", {"resource", "max", "damage_per"})
    rng = d.get("range", [0.0, 0.0])
    return Ability(
        id=d["id"],
        name=d.get("name", d["id"].replace("_", " ").title()),
        key=str(d.get("key", "")),
        cost=_num_map(d.get("cost", {}), where + ".cost"),
        gain=_num_map(d.get("gain", {}), where + ".gain"),
        set=_num_map(d.get("set", {}), where + ".set"),
        cooldown=float(d.get("cooldown", 0)),
        cooldown_group=d.get("cooldown_group", ""),
        gcd=None if d.get("gcd") is None else float(d["gcd"]),
        off_gcd=bool(d.get("off_gcd", False)),
        cast_time=float(d.get("cast_time", 0)),
        finisher=d.get("finisher", ""),
        damage=_damage(d["damage"], where + ".damage") if "damage" in d else None,
        dot=_dot(d["dot"], where + ".dot") if "dot" in d else None,
        extra=Extra(extra["resource"], float(extra["max"]), float(extra["damage_per"])) if extra else None,
        buff=d.get("buff", ""),
        debuff=d.get("debuff", ""),
        consumes_buff=d.get("consumes_buff", ""),
        next_swing=bool(d.get("next_swing", False)),
        gap_closer=bool(d.get("gap_closer", False)),
        range_min=float(rng[0]),
        range_max=float(rng[1]),
        req_behind=bool(req.get("behind", False)),
        req_target_health_below=float(req.get("target_health_below", 0)),
        req_target_health_above=float(req.get("target_health_above", 0)),
        req_buff=req.get("buff", ""),
        req_no_buff=req.get("no_buff", ""),
    )


def _validate(s: Spec) -> None:
    res = {r.id for r in s.resources}
    buffs = {b.id for b in s.buffs}
    debuffs = {b.id for b in s.debuffs}
    abilities = [a.id for a in s.abilities]
    if len(set(abilities)) != len(abilities):
        raise SpecError("duplicate ability ids")
    for r in s.resources:
        if r.kind not in RESOURCE_KINDS:
            raise SpecError(f"resource {r.id}: kind must be one of {RESOURCE_KINDS}")
    for b in s.buffs + s.debuffs:
        for r in b.cost_mult:
            if r != "all" and r not in res:
                raise SpecError(f"aura {b.id}: cost_mult references unknown resource {r!r}")
        for r in b.regen_mult:
            if r not in res:
                raise SpecError(f"aura {b.id}: regen_mult references unknown resource {r!r}")
        if b.consumed_by.startswith("ability:") and b.consumed_by[8:] not in abilities:
            raise SpecError(f"aura {b.id}: consumed_by references unknown ability {b.consumed_by[8:]!r}")
    for p in s.procs:
        if p.trigger not in PROC_TRIGGERS and not (p.trigger.startswith("ability:") and p.trigger[8:] in abilities):
            raise SpecError(f"proc {p.id}: trigger must be one of {PROC_TRIGGERS} or ability:<id>")
        if p.buff and p.buff not in buffs:
            raise SpecError(f"proc {p.id}: unknown buff {p.buff!r}")
        if p.debuff and p.debuff not in debuffs:
            raise SpecError(f"proc {p.id}: unknown debuff {p.debuff!r}")
        if p.requires_buff and p.requires_buff not in buffs:
            raise SpecError(f"proc {p.id}: unknown buff {p.requires_buff!r}")
        for r in p.gain:
            if r not in res:
                raise SpecError(f"proc {p.id}: gain references unknown resource {r!r}")
    for a in s.abilities:
        for m, name in ((a.cost, "cost"), (a.gain, "gain"), (a.set, "set")):
            for r in m:
                if r not in res:
                    raise SpecError(f"ability {a.id}: {name} references unknown resource {r!r}")
        if a.finisher and a.finisher not in res:
            raise SpecError(f"ability {a.id}: finisher references unknown resource {a.finisher!r}")
        if a.extra and a.extra.resource not in res:
            raise SpecError(f"ability {a.id}: extra references unknown resource {a.extra.resource!r}")
        for b, name in ((a.buff, "buff"), (a.consumes_buff, "consumes_buff"), (a.req_buff, "requires.buff"),
                        (a.req_no_buff, "requires.no_buff")):
            if b and b not in buffs:
                raise SpecError(f"ability {a.id}: {name} references unknown buff {b!r}")
        if a.debuff and a.debuff not in debuffs:
            raise SpecError(f"ability {a.id}: unknown debuff {a.debuff!r}")
        if a.next_swing and (a.cast_time or a.gap_closer or a.aoe):
            raise SpecError(f"ability {a.id}: next_swing abilities can't have a cast time, AoE, or gap closer")
    for aid, _ in s.apl:
        if aid not in abilities:
            raise SpecError(f"apl references unknown ability {aid!r}")


def load_spec(path: str | Path = "feral") -> Spec:
    p = Path(path)
    if not p.suffix:
        p = SPECS_DIR / f"{p}.toml"
    raw = tomllib.loads(p.read_text())
    _take(raw, p.name, {"class", "combat", "weapons", "resources", "buffs", "debuffs", "procs", "abilities", "teacher", "apl"})
    cls = _take(raw["class"], "class", {"name", "spec", "notes"})
    combat = _take(raw["combat"], "combat", {"gcd", "crit_chance", "crit_multiplier", "melee_range", "run_speed", "miss_chance",
                                             "dodge_chance", "glancing_chance", "glancing_multiplier", "dual_wield_miss"})
    weapons = _take(raw["weapons"], "weapons", {"main_hand", "off_hand"})
    teacher = _take(raw.get("teacher", {}), "teacher", {"resource_value", "reference_dps"})

    resources = []
    for rid, r in raw["resources"].items():
        _take(r, f"resources.{rid}", {"kind", "max", "start", "regen_per_sec", "tick", "per_white_damage", "on_target"})
        tick = r.get("tick", {})
        resources.append(Resource(
            id=rid,
            kind=r["kind"],
            max=float(r["max"]),
            start=float(r.get("start", r["max"] if r["kind"] in ("energy", "mana") else 0)),
            regen_per_sec=float(r.get("regen_per_sec", 0)),
            tick_amount=float(tick.get("amount", 0)),
            tick_interval=float(tick.get("interval", 0)),
            per_white_damage=float(r.get("per_white_damage", 0)),
            on_target=bool(r.get("on_target", False)),
        ))

    procs = []
    for i, pr in enumerate(raw.get("procs", [])):
        where = f"procs[{i}]"
        _take(pr, where, {"id", "trigger", "chance", "ppm", "buff", "debuff", "gain", "extra_attacks", "damage", "dot", "requires_buff"})
        procs.append(Proc(
            id=pr.get("id", f"proc{i}"),
            trigger=pr["trigger"],
            chance=float(pr.get("chance", 0)),
            ppm=float(pr.get("ppm", 0)),
            buff=pr.get("buff", ""),
            debuff=pr.get("debuff", ""),
            gain=_num_map(pr.get("gain", {}), where + ".gain"),
            extra_attacks=int(pr.get("extra_attacks", 0)),
            damage=_damage(pr["damage"], where + ".damage") if "damage" in pr else None,
            dot=_dot(pr["dot"], where + ".dot") if "dot" in pr else None,
            requires_buff=pr.get("requires_buff", ""),
        ))

    oh = weapons.get("off_hand")
    spec = Spec(
        class_name=cls["name"],
        spec_name=cls["spec"],
        notes=cls.get("notes", ""),
        gcd=float(combat.get("gcd", 1.5)),
        crit_chance=float(combat.get("crit_chance", 0.2)),
        crit_multiplier=float(combat.get("crit_multiplier", 2.0)),
        melee_range=float(combat.get("melee_range", 5.0)),
        run_speed=float(combat.get("run_speed", 7.0)),
        miss_chance=float(combat.get("miss_chance", 0)),
        dodge_chance=float(combat.get("dodge_chance", 0)),
        glancing_chance=float(combat.get("glancing_chance", 0)),
        glancing_multiplier=float(combat.get("glancing_multiplier", 0.95)),
        dual_wield_miss=float(combat.get("dual_wield_miss", 0)),
        main_hand=Weapon(float(weapons["main_hand"]["damage"]), float(weapons["main_hand"]["speed"])),
        off_hand=Weapon(float(oh["damage"]), float(oh["speed"])) if oh else None,
        resources=resources,
        buffs=[_aura(k, v, f"buffs.{k}") for k, v in raw.get("buffs", {}).items()],
        debuffs=[_aura(k, v, f"debuffs.{k}") for k, v in raw.get("debuffs", {}).items()],
        procs=procs,
        abilities=[_ability(a, i) for i, a in enumerate(raw["abilities"])],
        resource_value=_num_map(teacher.get("resource_value", {}), "teacher.resource_value"),
        reference_dps=float(teacher.get("reference_dps", 700)),
        apl=[(e["use"], e.get("if", "True")) for e in raw.get("apl", [])],
    )
    _validate(spec)
    return spec


def available_specs() -> list[str]:
    return sorted(p.stem for p in SPECS_DIR.glob("*.toml"))

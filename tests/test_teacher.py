import copy
import random
import statistics

import numpy as np
import pytest

from stim.policies import AplPolicy
from stim.scenarios import make_scenario
from stim.sim import POOL, TANK_GAP, EventPlan, Sim, rules_for
from stim.spec import load_spec
from stim.teacher import Teacher, TeacherConfig, state_value, summarize


def dummy(spec, hp=1e7, seed=0):
    sim = Sim(spec, random.Random(seed), t_max=300.0)
    sim.add_enemy("Dummy", 0.0, TANK_GAP, hp, 0.0, boss=True)
    return sim


def pool_until(sim, t):
    while sim.t < t and not sim.done:
        sim.step((POOL, -1))


def deterministic(name):
    spec = copy.deepcopy(load_spec(name))
    spec.crit_chance = 0.0
    spec.miss_chance = spec.dodge_chance = spec.glancing_chance = spec.dual_wield_miss = 0.0
    for p in spec.procs:
        p.chance = p.ppm = 0.0
    return spec


def mid_fight(name, kind="boss", seed=2, steps=20):
    spec = load_spec(name)
    sim = make_scenario(spec, kind, random.Random(seed))
    policy = AplPolicy(spec)
    for _ in range(steps):
        sim.step(policy(sim, sim.legal_options()))
    return spec, sim, policy


# ---------------------------------------------------------------------- engine support


def test_reseeded_clones_share_luck_across_different_choices():
    spec = load_spec("feral")
    sim = dummy(spec)
    pool_until(sim, 1.0)
    a, b = sim.clone(seed=7), sim.clone(seed=7)
    a.step((spec.ability_index("shifting_power"), -1))  # no damage: every swing after it lines up
    pool_until(a, 6.0)
    pool_until(b, 6.0)
    white = len(sim.dmg_by) - 1
    assert a.t == b.t and a.dmg_by[white] == b.dmg_by[white] > 0


def test_clone_without_seed_replays_the_same_rolls():
    spec, sim, policy = mid_fight("rogue_combat")
    a, b = sim.clone(), sim.clone()
    for c in (a, b):
        while not c.done:
            c.step(policy(c, c.legal_options()))
    assert a.damage == b.damage and a.t == b.t


def test_expected_damage_equals_damage_without_randomness():
    spec = deterministic("feral")
    sim = make_scenario(spec, "trash", random.Random(1))
    policy = AplPolicy(spec)
    while not sim.done:
        sim.step(policy(sim, sim.legal_options()))
    assert sim.ev_damage == pytest.approx(sim.damage, rel=1e-9)


def test_expected_damage_tracks_damage_on_average():
    spec = load_spec("feral")
    ratios = []
    for seed in range(100):
        sim = make_scenario(spec, "boss", random.Random(seed))
        policy = AplPolicy(spec)
        while not sim.done:
            sim.step(policy(sim, sim.legal_options()))
        ratios.append(sim.ev_damage / sim.damage - 1.0)
    assert abs(statistics.mean(ratios)) < 0.02


def test_event_resampling_knows_how_long_ago_the_last_event_was():
    plan = EventPlan(knockback=(18.0, 35.0))
    rng = random.Random(0)
    overdue = [plan.sample(rng, 30.0, 100.0, {"knockback": 0.0})[0][0] for _ in range(200)]
    assert all(30.0 <= t <= 35.0 for t in overdue)  # 30 s since the last one: due within 5 s
    recent = [plan.sample(rng, 30.0, 100.0, {"knockback": 25.0})[0][0] for _ in range(200)]
    assert all(43.0 <= t <= 60.0 for t in recent)
    sim = dummy(load_spec("feral"))
    pool_until(sim, 2.0)
    sim._event("knockback", (15.0,))
    assert sim.last_event["knockback"] == sim.t


def test_randomized_specs_get_their_own_rules():
    spec = load_spec("feral")
    rules_for(spec)  # the base spec has been simulated, so its compiled rules are cached
    r = spec.randomized(random.Random(1))
    shred = r.ability_index("shred")
    R = rules_for(r)
    assert R.cost[shred] == [(R.res_index["energy"], r.abilities[shred].cost["energy"])]


# ---------------------------------------------------------------------- terminal value


def test_state_value_counts_dot_ticks_that_land_before_death():
    spec = deterministic("feral")
    sim = dummy(spec)
    boss = sim.enemies[0]
    before = state_value(sim, sim.t_max)
    sim._apply_dot(boss, spec.ability_index("rip"), 6, 2.0, 200.0, "bleed")
    assert state_value(sim, sim.t_max) - before == pytest.approx(1200.0)
    boss.loss_rate = boss.hp / 5.0  # dies in 5 s: only the ticks at +2 s and +4 s land
    dying = state_value(sim, sim.t_max)
    del boss.dots[spec.ability_index("rip")]
    assert dying - state_value(sim, sim.t_max) == pytest.approx(400.0)


def test_unused_long_cooldowns_count_at_the_horizon():
    spec = load_spec("feral")
    sim = dummy(spec)
    tea = spec.ability_index("thistle_tea")
    ready = state_value(sim, sim.t_max)
    sim.cd_until[tea] = sim.t + 300.0
    assert ready - state_value(sim, sim.t_max) == pytest.approx(100 * spec.resource_value["energy"])


def test_maintained_auras_are_worth_their_upkeep():
    spec = load_spec("warrior_fury")
    sim = dummy(spec)
    shout = sim.R.buff_index["battle_shout"]
    down = state_value(sim, sim.t_max)
    sim.buff_until[shout] = sim.t + 100.0
    upkeep = 10 * spec.resource_value["rage"] / 120.0  # 10 rage per 120 s, far below its DPS value
    assert state_value(sim, sim.t_max) - down == pytest.approx(100.0 * upkeep)


# ---------------------------------------------------------------------- labels


def test_soft_targets_are_normalized_and_never_hard():
    q = np.array([[100.0, 110, 90, 105], [0, 0, 0, 0], [99, 108, 92, 104]])
    s = summarize(q, 700.0, TeacherConfig())
    assert s["best"] == 0 and s["gap"][0] == 0.0
    assert s["gap"][1] == pytest.approx(101.25)
    assert s["soft"].sum() == pytest.approx(1.0)
    assert (s["soft"] > 0).all() and s["soft"].argmax() == 0
    assert s["soft"][2] > 0.9 * s["soft"][0]  # a near-tie stays a near-tie
    assert s["stakes_level"] == 0


def test_identical_options_get_identical_values():
    spec, sim, policy = mid_fight("feral")
    o = sim.legal_options()[-1]
    label = Teacher(policy, TeacherConfig(samples=4)).evaluate(sim, random.Random(0), [o, o])
    assert np.array_equal(label.q[0], label.q[1])
    assert label.soft[0] == pytest.approx(0.5)


def test_labels_ignore_the_real_future():
    spec, sim, policy = mid_fight("feral", "boss_adds", seed=4, steps=15)
    teacher = Teacher(policy, TeacherConfig(samples=4))
    a = teacher.evaluate(sim, random.Random(1))
    other = sim.clone(seed=123)  # different combat rolls ahead...
    other.events = other.events[: other.ev_i] + [(sim.t + 1.0, "knockback", (15.0,)), (sim.t + 2.0, "turn", (3.0,))]
    b = teacher.evaluate(other, random.Random(1))  # ...and a different script: the label can't change
    assert a.options == b.options and np.array_equal(a.q, b.q)


def test_teacher_spends_capped_energy_instead_of_pooling():
    spec = load_spec("feral")
    sim = dummy(spec)
    pool_until(sim, 1.0)
    sim.res[sim.R.res_index["energy"]] = 100.0
    label = Teacher(AplPolicy(spec), TeacherConfig(samples=48)).evaluate(sim, random.Random(0))
    pool = label.options.index((POOL, -1))
    assert label.best != pool and label.gap[pool] > label.se[pool]
    assert 0.0 <= label.survive <= 1.0 and label.stakes > 0.0


def test_batched_policies_decide_for_many_rollouts_at_once():
    spec, sim, policy = mid_fight("warrior_fury")

    class Batched:
        def __init__(self):
            self.sizes = []

        def batch(self, sims, opts):
            self.sizes.append(len(sims))
            return [policy(s, o) for s, o in zip(sims, opts)]

    batched = Batched()
    a = Teacher(policy, TeacherConfig(samples=4)).evaluate(sim, random.Random(3))
    b = Teacher(batched, TeacherConfig(samples=4)).evaluate(sim, random.Random(3))
    assert np.array_equal(a.q, b.q)
    assert max(batched.sizes) == 4 * len(a.options)

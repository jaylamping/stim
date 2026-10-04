import copy
import random

import pytest

from stim.policies import AplPolicy, GreedyPolicy, RandomPolicy
from stim.scenarios import SCENARIOS, make_scenario
from stim.sim import POOL, TANK_GAP, Sim
from stim.spec import available_specs, load_spec


def dummy(spec, hp=1e7, seed=0):
    sim = Sim(spec, random.Random(seed), t_max=300.0)
    sim.add_enemy("Dummy", 0.0, TANK_GAP, hp, 0.0, boss=True)
    return sim


def deterministic(name):
    spec = copy.deepcopy(load_spec(name))
    spec.crit_chance = 0.0
    spec.miss_chance = spec.dodge_chance = spec.glancing_chance = spec.dual_wield_miss = 0.0
    for p in spec.procs:
        p.chance = p.ppm = 0.0
    return spec


def pool_until(sim, t):
    while sim.t < t and not sim.done:
        sim.step((POOL, -1))


def legal_ids(sim):
    return {(sim.spec.abilities[a].id if a >= 0 else "pool", tid) for a, tid in sim.legal_options()}


@pytest.mark.parametrize("name", available_specs())
@pytest.mark.parametrize("kind", SCENARIOS)
def test_every_spec_runs_every_scenario(name, kind):
    spec = load_spec(name)
    for policy in (RandomPolicy(random.Random(0)), GreedyPolicy(), AplPolicy(spec)):
        sim = make_scenario(spec, kind, random.Random(3))
        steps = 0
        while not sim.done:
            opts = sim.legal_options()
            assert opts[0] == (POOL, -1)
            sim.step(policy(sim, opts))
            steps += 1
            assert steps < 10000
        assert sim.damage > 0


@pytest.mark.parametrize("name", available_specs())
def test_priority_list_beats_random(name):
    spec = load_spec(name)

    def mean_dps(policy):
        total = 0.0
        for ep in range(12):
            sim = make_scenario(spec, "boss", random.Random(100 + ep))
            while not sim.done:
                sim.step(policy(sim, sim.legal_options()))
            total += sim.damage / sim.t
        return total / 12

    assert mean_dps(AplPolicy(spec)) > mean_dps(RandomPolicy(random.Random(1)))


def test_energy_ticks():
    spec = deterministic("feral")
    sim = dummy(spec)
    e = sim.R.res_index["energy"]
    sim.res[e], sim.next_tick[e] = 0.0, 0.5
    pool_until(sim, 2.6)
    assert sim.res[e] == 40.0


def test_combo_points_live_on_the_target():
    spec = deterministic("feral")
    sim = dummy(spec)
    second = sim.add_enemy("Second", 1.0, TANK_GAP, 1e7, 0.0)
    pool_until(sim, 0.5)
    first = sim.enemies[0]
    cp, bite = sim.R.res_index["combo_points"], spec.ability_index("primal_bite")
    sim.step((bite, first.id))
    assert (sim.res[cp], sim.res_target[cp]) == (1.0, first.id)
    sim.res[sim.R.res_index["energy"]] = 100.0
    sim.step((bite, second.id))
    assert (sim.res[cp], sim.res_target[cp]) == (1.0, second.id)
    assert ("rip", second.id) in legal_ids(sim) and ("rip", first.id) not in legal_ids(sim)


def test_finisher_scales_with_points_and_consumes_them():
    spec = deterministic("feral")
    sim = dummy(spec)
    boss = sim.enemies[0]
    cp, rip = sim.R.res_index["combo_points"], spec.ability_index("rip")
    sim.res[cp], sim.res_target[cp] = 5.0, boss.id
    sim.step((rip, boss.id))
    assert sim.res[cp] == 0.0
    assert boss.dots[rip][2] == pytest.approx(30 + 34 * 5)


def test_shred_needs_behind_and_turns_break_it():
    sim = dummy(deterministic("feral"))
    pool_until(sim, 0.5)
    boss = sim.enemies[0]
    assert ("shred", boss.id) in legal_ids(sim)
    boss.turn_until = sim.t + 3.0
    ids = legal_ids(sim)
    assert ("shred", boss.id) not in ids and ("primal_bite", boss.id) in ids


def test_gap_closer_lands_behind_the_target():
    spec = deterministic("feral")
    sim = dummy(spec)
    pool_until(sim, 0.5)
    boss = sim.enemies[0]
    sim._event("knockback", (15.0,))
    pool_until(sim, sim.t + 0.7)
    assert ("feral_charge", boss.id) in legal_ids(sim)
    sim.step((spec.ability_index("feral_charge"), boss.id))
    assert sim.dist(boss) <= spec.melee_range and sim.behind(boss)


def test_rage_comes_from_white_hits_and_heroic_strike_spends_it():
    spec = deterministic("warrior_fury")
    sim = dummy(spec)
    rage, hs = sim.R.res_index["rage"], spec.ability_index("heroic_strike")
    assert sim.res[rage] == 0.0
    pool_until(sim, 3.0)
    assert sim.res[rage] > 0.0
    sim.res[rage] = 100.0
    sim.step((hs, -1))
    assert sim.queued == hs
    pool_until(sim, sim.swing[0] + 0.01)
    assert sim.queued == -1 and sim.dmg_by[hs] > 0


def test_clone_is_independent():
    spec = load_spec("rogue_combat")
    sim = make_scenario(spec, "trash", random.Random(5))
    pool_until(sim, 2.0)
    before = (sim.t, sim.damage, [e.hp for e in sim.enemies], list(sim.res))
    c = sim.clone()
    c.rng = random.Random(9)
    policy = GreedyPolicy()
    while not c.done:
        c.step(policy(c, c.legal_options()))
    assert (sim.t, sim.damage, [e.hp for e in sim.enemies], list(sim.res)) == before


def test_same_seed_same_fight():
    spec = load_spec("feral")

    def run():
        sim = make_scenario(spec, "boss_adds", random.Random(42))
        policy = AplPolicy(spec)
        while not sim.done:
            sim.step(policy(sim, sim.legal_options()))
        return sim.t, sim.damage

    assert run() == run()


def test_bad_spec_fields_are_rejected(tmp_path):
    from stim.spec import SpecError

    src = (load_spec.__globals__["SPECS_DIR"] / "feral.toml").read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(src.replace("weapon_coeff", "weapon_cof").replace("weapon = 2.25", "wepon = 2.25"))
    with pytest.raises(SpecError, match="wepon"):
        load_spec(bad)

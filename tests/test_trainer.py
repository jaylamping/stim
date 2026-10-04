import json

import torch

from stim.model import ModelConfig, StimNet
from stim.sim import POOL
from stim.student import StudentPolicy
from stim.trainer.session import IDLE_GRACE, QUEUE_WINDOW, Session


def coach():
    torch.manual_seed(0)
    return StudentPolicy(StimNet(ModelConfig(d=32, layers=1, readout_blocks=1)))


def run_until_ready(s, limit=5.0):
    while s.sim.gcd_until > s.sim.t and limit > 0:
        s.tick(0.05)
        limit -= 0.05


def key_of(s, ability_id):
    return s.spec.abilities[s.spec.ability_index(ability_id)].key


def test_a_legal_press_acts_and_is_recorded():
    s = Session("warrior_fury", "boss", seed=1, student=coach())
    s.tick(0.5)  # walk into melee range
    s.sim.res[s.sim.R.res_index["rage"]] = 50.0
    assert s.press(key_of(s, "bloodthirst")) == ""
    assert s.sim.gcd_until > s.sim.t
    d = s.decisions[-1]
    assert d.chosen == (s.spec.ability_index("bloodthirst"), s.sim.target)
    assert d.chosen in d.options and d.advice in d.options


def test_presses_near_the_end_of_the_gcd_are_queued():
    s = Session("feral", "boss", seed=2, student=coach())
    s.tick(0.5)
    s.sim.res[s.sim.R.res_index["energy"]] = 100.0
    assert s.press(key_of(s, "claw")) == ""
    while s.sim.gcd_until - s.sim.t > QUEUE_WINDOW:
        s.tick(0.05)
    before = len(s.decisions)
    assert s.press(key_of(s, "claw")) == ""
    assert s.queued is not None
    run_until_ready(s)
    s.tick(0.05)
    assert s.queued is None and len(s.decisions) == before + 1


def test_refusals_say_why():
    s = Session("feral", "boss", seed=3)
    s.tick(0.5)
    s.sim.res[s.sim.R.res_index["energy"]] = 0.0
    assert "energy" in s.press(key_of(s, "shred"))
    s.sim.res[s.sim.R.res_index["energy"]] = 100.0
    assert "combo points" in s.press(key_of(s, "rip"))
    assert s.press(key_of(s, "claw")) == ""
    assert s.press(key_of(s, "claw")) == "not ready yet"


def test_idling_with_something_usable_is_a_decision_to_wait():
    s = Session("warrior_fury", "boss", seed=4)
    s.sim.res[s.sim.R.res_index["rage"]] = 60.0
    t = 0.0
    while t < IDLE_GRACE + 1.0:
        s.tick(0.05)
        t += 0.05
    assert any(d.chosen == (POOL, -1) for d in s.decisions)


def test_state_is_json_and_review_grades_decisions():
    s = Session("feral", "trash", seed=5, student=coach())
    for _ in range(80):
        s.tick(0.1)
        advice = s.advice()
        if advice and s.sim.gcd_until <= s.sim.t:
            s._try(advice["option"])  # play the coach's picks
    json.dumps(s.state())
    assert s.decisions
    review = s.review(samples=4)
    assert review["decisions"] == len(s.decisions)
    assert all(w["regret"] >= 0 for w in review["worst"])
    assert 0.0 <= review["advice_followed"] <= 1.0

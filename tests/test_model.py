import math
import random

import numpy as np
import pytest
import torch

from stim import features as F
from stim.model import Batch, ModelConfig, StimNet
from stim.policies import AplPolicy
from stim.scenarios import SCENARIOS, make_scenario
from stim.spec import available_specs, load_spec
from stim.student import StudentPolicy, load_student, save_checkpoint


def states(name, kind, n=12, seed=3):
    spec = load_spec(name)
    sim = make_scenario(spec, kind, random.Random(seed))
    policy = AplPolicy(spec)
    out = []
    while not sim.done and len(out) < n:
        opts = sim.legal_options()
        if len(opts) > 1:
            out.append(sim.clone())
        sim.step(policy(sim, opts))
    return out


@pytest.mark.parametrize("name", available_specs())
@pytest.mark.parametrize("kind", SCENARIOS)
def test_tokens_are_finite_and_well_formed(name, kind):
    for sim in states(name, kind):
        opts = sim.legal_options()
        tok = F.tokens(sim, opts)
        assert tok.player.shape == (F.PLAYER,)
        assert tok.res.shape[1] == F.RESOURCE and tok.aura.shape[1] == F.AURA
        assert tok.abil.shape == (sim.R.n_abilities, F.ABILITY)
        assert tok.enemy.shape == (min(len(sim.enemies), F.MAX_ENEMIES), F.ENEMY)
        assert tok.opt.shape == (len(opts), F.OPTION)
        for a in (tok.player, tok.res, tok.aura, tok.abil, tok.enemy, tok.opt):
            assert np.isfinite(a).all()
        assert ((tok.opt_abil >= -1) & (tok.opt_abil < sim.R.n_abilities)).all()
        assert ((tok.opt_enemy >= -1) & (tok.opt_enemy < tok.enemy.shape[0])).all()
        for (ai, tid), ab, en in zip(opts, tok.opt_abil, tok.opt_enemy):
            assert ab == ai
            if tid >= 0:
                assert tok.enemy[en, 8] == float(tid == sim.target)  # column 8: "is the current target"


def rotated(sim, angle):
    c, s = math.cos(angle), math.sin(angle)
    r = sim.clone()

    def rot(x, y):
        return c * x - s * y, s * x + c * y

    r.px, r.py = rot(r.px, r.py)
    r.tank_x, r.tank_y = rot(r.tank_x, r.tank_y)
    for e in r.enemies:
        e.x, e.y = rot(e.x, e.y)
        e.fx, e.fy = rot(e.fx, e.fy)
    return r


def test_tokens_do_not_change_when_the_fight_is_rotated():
    for sim in states("feral", "trash", n=6):
        a, b = F.tokens(sim), F.tokens(rotated(sim, 2.1))
        for name in ("player", "res", "aura", "abil", "enemy", "opt"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-4)


def test_padded_options_get_no_probability_and_rows_normalize():
    toks = [F.tokens(s) for name in available_specs() for s in states(name, "trash", n=4)]
    b = Batch(toks)
    p = StimNet()(b)["choice"].softmax(-1)
    assert torch.allclose(p.sum(-1), torch.ones(len(toks)))
    assert (p[~b.mask["opt"]] == 0).all()


def test_batched_decisions_match_one_at_a_time(tmp_path):
    torch.manual_seed(0)
    model = StimNet(ModelConfig(d=32, layers=1, readout_blocks=1))
    sims = states("rogue_combat", "boss_adds", n=8)
    opts = [s.legal_options() for s in sims]
    policy = StudentPolicy(model)
    assert policy.batch(sims, opts) == [policy(s, o) for s, o in zip(sims, opts)]
    path = tmp_path / "m.pt"
    save_checkpoint(path, model)
    assert load_student(path).batch(sims, opts) == policy.batch(sims, opts)
    for o, choice in zip(opts, policy.batch(sims, opts)):
        assert choice in o

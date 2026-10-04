"""Train the in-game addon's small policy on teacher-labeled states, using only what the addon's tracker
knew at each one.

    uv run python scripts/train_addon.py --data runs/addon0 --spec feral --out runs/addon0/addon_feral.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from stim.addon.policy import AddonNet, addon_examples, save_addon
from stim.addon.tracker import Tracker
from stim.data import load_episodes
from stim.spec import load_spec


def tensors(exs):
    x = torch.from_numpy(np.stack([e.obs for e in exs]))
    m = torch.from_numpy(np.stack([e.possible for e in exs]))
    y = torch.from_numpy(np.stack([e.soft for e in exs]))
    return x, m, y


def loss_fn(logits, mask, soft):
    logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    return (-(soft * logp).sum(-1) + 0.5 * ((p - soft) ** 2).sum(-1)).mean()


@torch.no_grad()
def evaluate(net, exs) -> dict:
    x, m, y = tensors(exs)
    logits = net(x, m)
    order = torch.argsort(logits, dim=-1, descending=True).numpy()
    rows = defaultdict(list)
    for e, ranked in zip(exs, order):
        # like a player following the addon: the first suggestion the game accepts
        a = next(int(i) for i in ranked if np.isfinite(e.gap[i]))
        apl = e.gap[e.apl] if e.apl >= 0 and np.isfinite(e.gap[e.apl]) else np.nan
        legal_gaps = e.gap[np.isfinite(e.gap)]
        rows[e.kind].append((e.gap[a], apl, legal_gaps.mean(), float(a == int(np.argmax(e.soft)))))
    out = {"loss": loss_fn(logits, m, y).item()}
    allrows = [r for v in rows.values() for r in v]
    a = np.array(allrows, dtype=np.float64)
    out.update(regret=float(a[:, 0].mean()), regret_apl=float(np.nanmean(a[:, 1])), regret_random=float(a[:, 2].mean()),
               top1=float(a[:, 3].mean()))
    out["per_kind"] = {k: np.nanmean(np.array(v, dtype=np.float64), axis=0)[:3].round(4).tolist() for k, v in rows.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    episodes = [ep for d in args.data for ep in load_episodes(d) if ep.spec_name == args.spec]
    exs = addon_examples(episodes)
    eps = sorted({e.episode for e in exs})
    held = set(random.Random(args.seed).sample(eps, max(1, len(eps) // 10)))
    train = [e for e in exs if e.episode not in held]
    val = [e for e in exs if e.episode in held]
    x, _, _ = tensors(train)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-3)
    tracker = Tracker(load_spec(args.spec))
    net = AddonNet(tracker.width, tracker.A + 1, args.hidden, mean, std)
    print(f"{args.spec}: {len(train)} train / {len(val)} val states, {sum(p.numel() for p in net.parameters())} weights")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    best = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        rng.shuffle(train)
        for i in range(0, len(train), args.batch):
            xb, mb, yb = tensors(train[i : i + args.batch])
            loss = loss_fn(net(xb, mb), mb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        v = evaluate(net, val)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val loss {v['loss']:.4f}  regret {v['regret']:.4f}s  "
                  f"(apl {v['regret_apl']:.4f}, random {v['regret_random']:.4f})  top1 {v['top1']:.3f}", flush=True)
        if best is None or v["regret"] < best["regret"]:
            best = dict(v, epoch=epoch)
            save_addon(args.out, net, args.spec, tracker.features_layout(), {"val": best})
    print(f"best epoch {best['epoch']}: regret {best['regret']:.4f}s (apl {best['regret_apl']:.4f}); "
          f"per scenario [addon, apl, random]: {json.dumps(best['per_kind'])}")


if __name__ == "__main__":
    main()

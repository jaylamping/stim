"""Train the decision model on teacher-labeled states.

Loss (RLCD-style, never hard labels): soft-target NLL + 0.5 Brier on the choice; the confidence head
learns how much teacher mass sits on the model's own top option (a stop-gradient target); BCE + Brier
on survival; ordinal NLL on stakes. Validation holds out whole episodes and reports regret against
the teacher, next to what the baseline policies' choices would have cost on the same states.

    uv run python scripts/train.py --data runs/gen0 --out runs/gen0/student.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from stim.dataset import Example, examples_from_run, split_by_episode
from stim.model import STAKES_LEVELS, Batch, ModelConfig, StimNet, n_params
from stim.student import save_checkpoint


class Targets:
    def __init__(self, exs: list[Example], m: int, device):
        n = len(exs)
        soft = np.zeros((n, m), dtype=np.float32)
        gap = np.zeros((n, m), dtype=np.float32)
        for i, ex in enumerate(exs):
            soft[i, : len(ex.soft)] = ex.soft
            gap[i, : len(ex.gap)] = ex.gap
        self.soft = torch.from_numpy(soft).to(device)
        self.gap = torch.from_numpy(gap).to(device)
        self.survive = torch.tensor([ex.survive for ex in exs], dtype=torch.float32, device=device)
        self.level = torch.tensor([ex.level for ex in exs], dtype=torch.long, device=device)
        self.best = torch.tensor([ex.best for ex in exs], dtype=torch.long, device=device)


def losses(out: dict[str, torch.Tensor], tg: Targets, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = out["choice"]
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    nll = -(tg.soft * logp).sum(-1)
    brier = ((p - tg.soft) ** 2).sum(-1)
    top = logits.argmax(dim=-1, keepdim=True)
    agree = tg.soft.gather(1, top).squeeze(1)  # teacher mass on our top option: data, so no gradient
    conf = Fn.binary_cross_entropy_with_logits(out["conf"], agree, reduction="none")
    noul_p = torch.sigmoid(out["noul"])
    noul = Fn.binary_cross_entropy_with_logits(out["noul"], tg.survive, reduction="none") + (noul_p - tg.survive) ** 2
    y = (tg.level[:, None] > torch.arange(STAKES_LEVELS - 1, device=mask.device)).float()
    ordinal = Fn.binary_cross_entropy_with_logits(out["score"], y, reduction="none").sum(-1)
    total = nll + 0.5 * brier + 0.2 * conf + 0.5 * noul + 0.3 * ordinal
    parts = {"nll": nll, "brier": brier, "conf": conf, "noul": noul, "ordinal": ordinal}
    return total.mean(), {k: v.detach() for k, v in parts.items()}


def batches(exs: list[Example], size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(exs)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [exs[j] for j in order[i : i + size]]


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - hit[sel].mean())
    return float(total)


@torch.no_grad()
def evaluate(model: StimNet, exs: list[Example], device, size: int = 512) -> dict:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    rows = []
    for chunk in batches(exs, size, False, random.Random(0)):
        b = Batch([ex.tok for ex in chunk], device)
        tg = Targets(chunk, b.x["opt"].shape[1], device)
        out = model(b)
        loss, parts = losses(out, tg, b.mask["opt"])
        sums["loss"] += loss.item() * len(chunk)
        for k, v in parts.items():
            sums[k] += v.sum().item()
        top = out["choice"].argmax(-1).cpu().numpy()
        pmax = out["choice"].softmax(-1).max(-1).values.cpu().numpy()
        conf = torch.sigmoid(out["conf"]).cpu().numpy()
        noul = torch.sigmoid(out["noul"]).cpu().numpy()
        cum = torch.sigmoid(out["score"]).cpu().numpy()  # P(level > k)
        level = (cum > 0.5).sum(-1)
        for i, ex in enumerate(chunk):
            rows.append((ex.spec_name, ex.kind, float(ex.gap[top[i]]), float(ex.gap[ex.baselines["apl"]]),
                         float(ex.gap[ex.baselines["greedy"]]), float(ex.gap.mean()), float(top[i] == ex.best),
                         float(ex.soft[top[i]]), float(conf[i]), float(pmax[i]), float(noul[i]), ex.survive,
                         float(level[i] == ex.level)))
    n = len(exs)
    a = np.array([r[2:] for r in rows], dtype=np.float64)
    res = {k: v / n for k, v in sums.items()}
    res.update(
        regret=a[:, 0].mean(), regret_apl=a[:, 1].mean(), regret_greedy=a[:, 2].mean(), regret_random=a[:, 3].mean(),
        top1=a[:, 4].mean(), agree=a[:, 5].mean(), ece_conf=ece(a[:, 6], a[:, 5]), ece_pmax=ece(a[:, 7], a[:, 5]),
        noul_brier=float(((a[:, 8] - a[:, 9]) ** 2).mean()), stakes_acc=a[:, 10].mean(),
    )
    per = defaultdict(list)
    for r in rows:
        per[r[0]].append(r[2:6])
    res["per_spec"] = {k: dict(zip(("student", "apl", "greedy", "random"), np.array(v).mean(0).round(4).tolist()))
                       for k, v in per.items()}
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    for f in dataclasses.fields(ModelConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    examples: list[Example] = []
    for i, d in enumerate(args.data):
        examples.extend(examples_from_run(d, episode_offset=i * 1_000_000))
    train, val = split_by_episode(examples, args.val_frac, args.seed)
    cfg = ModelConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ModelConfig)})
    model = StimNet(cfg).to(args.device)
    print(f"{len(train)} train / {len(val)} val states, {n_params(model)} parameters, device {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tl, tn = 0.0, 0
        for chunk in batches(train, args.batch, True, rng):
            b = Batch([ex.tok for ex in chunk], args.device)
            tg = Targets(chunk, b.x["opt"].shape[1], args.device)
            loss, _ = losses(model(b), tg, b.mask["opt"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(chunk)
            tn += len(chunk)
        v = evaluate(model, val, args.device)
        v.update(epoch=epoch, train_loss=tl / tn, seconds=round(time.perf_counter() - start))
        history.append(v)
        print(f"epoch {epoch:2d}  train {tl / tn:.4f}  val {v['loss']:.4f}  regret {v['regret']:.4f}s "
              f"(apl {v['regret_apl']:.4f}, greedy {v['regret_greedy']:.4f}, random {v['regret_random']:.4f})  "
              f"top1 {v['top1']:.3f}  ece {v['ece_conf']:.3f}  stakes {v['stakes_acc']:.3f}  [{v['seconds']}s]",
              flush=True)
        if best is None or v["regret"] < best["regret"]:
            best = v
            save_checkpoint(out_path, model, {"val": {k: x for k, x in v.items() if k != "per_spec"}, "args": vars(args)})
    (out_path.parent / (out_path.stem + "_history.json")).write_text(json.dumps(history, indent=1))
    print(f"best epoch {best['epoch']}: regret {best['regret']:.4f}s; per spec (student, apl, greedy, random):")
    for k, s in best["per_spec"].items():
        print(f"  {k:13s} {s}")


if __name__ == "__main__":
    main()

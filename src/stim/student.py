"""The trained decision model as a policy, plus saving and loading checkpoints."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from . import features as F
from .model import Batch, ModelConfig, StimNet
from .sim import Sim

Option = tuple[int, int]


class StudentPolicy:
    """Plays the model's top option. `batch` decides for many sims in one forward pass, which is how
    teacher rollouts use it."""

    name = "student"

    def __init__(self, model: StimNet, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def scores(self, sims: list[Sim], opts: list[list[Option]]) -> dict[str, torch.Tensor]:
        return self.model(Batch([F.tokens(s, o) for s, o in zip(sims, opts)], self.device))

    def batch(self, sims: list[Sim], opts: list[list[Option]]) -> list[Option]:
        out: list[Option | None] = [None] * len(sims)
        todo = []
        for i, o in enumerate(opts):
            if len(o) == 1:
                out[i] = o[0]
            else:
                todo.append(i)
        if todo:
            logits = self.scores([sims[i] for i in todo], [opts[i] for i in todo])["choice"]
            best = logits.argmax(dim=-1).tolist()
            for i, b in zip(todo, best):
                out[i] = opts[i][b]
        return out  # type: ignore[return-value]

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        return self.batch([sim], [opts])[0]


def save_checkpoint(path: str | Path, model: StimNet, extra: dict | None = None) -> None:
    torch.save({"config": dataclasses.asdict(model.cfg), "features": F.VERSION, "state": model.state_dict(),
                **(extra or {})}, path)


def load_student(path: str | Path, device: str = "cpu") -> StudentPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("features") != F.VERSION:
        raise ValueError(f"{path} was trained on features v{ckpt.get('features')}, this code makes v{F.VERSION}")
    model = StimNet(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    return StudentPolicy(model, device)

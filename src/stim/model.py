"""Jev-style decision model: a bidirectional encoder over entity tokens, learned query slots that read
it out, and typed heads.

- Choice: scores each legal option as q · option (option tokens are runtime-supplied, so the option
  count can vary), as a softmax over legal options
- Noul: will the current target still be alive in 12 s?
- Score: how much this decision matters (ordinal stakes level)
- Confidence: how much of the teacher's probability mass sits on the model's top option

Option tokens add the input embeddings of the ability and enemy tokens they point at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import features as F

TYPES = ("player", "res", "aura", "abil", "enemy", "opt")
WIDTH = {"player": F.PLAYER, "res": F.RESOURCE, "aura": F.AURA, "abil": F.ABILITY, "enemy": F.ENEMY, "opt": F.OPTION}
QUESTIONS = ("choice", "noul", "score", "conf")
STAKES_LEVELS = 4


@dataclass
class ModelConfig:
    d: int = 96
    heads: int = 4
    layers: int = 3
    slots: int = 4  # learned query slots per question
    readout_blocks: int = 2
    dropout: float = 0.0


class Batch:
    """Padded token arrays for a batch of states, as tensors."""

    def __init__(self, toks: list[F.Tokens], device: torch.device | str = "cpu"):
        n = len(toks)
        self.n = n
        self.x: dict[str, torch.Tensor] = {}
        self.mask: dict[str, torch.Tensor] = {}
        self.x["player"] = torch.from_numpy(np.stack([t.player for t in toks])).to(device)[:, None, :]
        self.mask["player"] = torch.ones(n, 1, dtype=torch.bool, device=device)
        for name in TYPES[1:]:
            arrs = [getattr(t, name) for t in toks]
            m = max(1, max(a.shape[0] for a in arrs))
            out = np.zeros((n, m, WIDTH[name]), dtype=np.float32)
            mask = np.zeros((n, m), dtype=bool)
            for i, a in enumerate(arrs):
                out[i, : a.shape[0]] = a
                mask[i, : a.shape[0]] = True
            self.x[name] = torch.from_numpy(out).to(device)
            self.mask[name] = torch.from_numpy(mask).to(device)
        m = self.x["opt"].shape[1]
        ab = np.full((n, m), -1, dtype=np.int64)
        en = np.full((n, m), -1, dtype=np.int64)
        for i, t in enumerate(toks):
            ab[i, : len(t.opt_abil)] = t.opt_abil
            en[i, : len(t.opt_enemy)] = t.opt_enemy
        self.opt_abil = torch.from_numpy(ab).to(device)
        self.opt_enemy = torch.from_numpy(en).to(device)


class ReadoutBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


class StimNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        d = cfg.d
        self.proj = nn.ModuleDict({k: nn.Linear(WIDTH[k], d) for k in TYPES})
        self.type_emb = nn.Parameter(torch.randn(len(TYPES), d) * 0.02)
        self.pool_emb = nn.Parameter(torch.randn(d) * 0.02)  # stands in for the ability of "wait"
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(len(QUESTIONS) * cfg.slots, d) * 0.02)
        self.readout = nn.ModuleList([ReadoutBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.readout_blocks)])
        self.readout_norm = nn.LayerNorm(d)
        self.choice_q = nn.Linear(d, d)
        self.choice_k = nn.Linear(d, d)
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.noul = nn.Linear(d, 1)
        self.score = nn.Linear(d, 1)
        self.score_cuts = nn.Parameter(torch.linspace(-1.0, 1.0, STAKES_LEVELS - 1))
        self.conf = nn.Linear(d, 1)

    def forward(self, b: Batch) -> dict[str, torch.Tensor]:
        d = self.cfg.d
        emb = {k: self.proj[k](b.x[k]) + self.type_emb[i] for i, k in enumerate(TYPES)}
        # options also carry the input embeddings of the ability and enemy they point at
        n = b.n
        idx = torch.arange(n, device=b.opt_abil.device)[:, None]
        ab = emb["abil"][idx, b.opt_abil.clamp(min=0)]
        ab = torch.where((b.opt_abil >= 0)[..., None], ab, self.pool_emb.expand_as(ab))
        en = emb["enemy"][idx, b.opt_enemy.clamp(min=0)] * (b.opt_enemy >= 0)[..., None]
        emb["opt"] = emb["opt"] + ab + en
        x = torch.cat([emb[k] for k in TYPES], dim=1)
        keep = torch.cat([b.mask[k] for k in TYPES], dim=1)
        h = self.final_norm(self.encoder(x, src_key_padding_mask=~keep))

        q = self.queries.expand(n, -1, -1)
        for block in self.readout:
            q = block(q, h, ~keep)
        q = self.readout_norm(q).view(n, len(QUESTIONS), self.cfg.slots, d).mean(dim=2)

        n_opt = b.x["opt"].shape[1]
        opt_h = h[:, -n_opt:]
        logits = torch.einsum("bd,bod->bo", self.choice_q(q[:, 0]), self.choice_k(opt_h)) / math.sqrt(d)
        logits = logits / self.log_temp.exp()
        logits = logits.masked_fill(~b.mask["opt"], float("-inf"))
        score = self.score(q[:, 2])  # (n, 1)
        return {
            "choice": logits,
            "noul": self.noul(q[:, 1]).squeeze(-1),
            "score": score - self.score_cuts.sort().values,  # cumulative logits for P(level > k)
            "conf": self.conf(q[:, 3]).squeeze(-1),
        }


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

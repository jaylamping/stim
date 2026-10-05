"""Write the Stim addon's data file (spec rules + trained weights) and its .toc.

    uv run python scripts/export_addon.py --spec feral --model runs/addon1/addon_feral.pt

Then copy addon/Stim into World of Warcraft/_classic_beta_/Interface/AddOns/ (or the live
Forever folder once it ships).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stim.addon.export import write_addon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="feral")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="addon/Stim")
    args = ap.parse_args()
    path = write_addon(args.spec, args.model, Path(args.out))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

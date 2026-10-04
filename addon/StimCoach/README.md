# StimCoach

Flashes your next three spells next to your character, Hekili-style, on WoW Forever. The recommendations
come from stim's simulation-trained policy, distilled into a small network that runs in the addon.
Every press is yours: StimCoach only shows icons.

Feral Druid (cat) for now.

## Install

Copy this folder (`StimCoach`, with all its `.lua` files and the `.toc`) into
`World of Warcraft/_classic_beta_/Interface/AddOns/` (the beta), or the live Forever AddOns folder once
it ships. Log in on your druid and target something hostile.

| Command | What it does |
| --- | --- |
| `/stim unlock`, then drag, then `/stim lock` | Move the icons |
| `/stim scale 1.2` | Resize |
| `/stim count 2` | Show 1-3 spells |
| `/stim toggle` | Show or hide |
| `/stim reset` | Default position and size |

The first icon pulses. It's greyed out while the spell is still a short wait away, for example while
you pool energy.

## How it works within Forever's rules

In combat, Forever hides energy, combo points, buffs, cooldown values and target health from addon
code. StimCoach doesn't try to read them. It works from what the game allows:
- **Your own casts.** It tracks cooldowns, the GCD, and your Rip, Rake and Faerie Fire timers on your
  target. It also tracks your Berserk window, builders since your last finisher, and an energy
  estimate.
- **Range checks and the clock.**
- **Resource values read before the pull.**

Energy matters most, so StimCoach precomputes a recommendation for each fifth of your energy bar.
The game itself decides which one to show: `UnitPowerPercent` with a step curve, drawn as the frame's
alpha, the same mechanism WeakAuras Forever's "Show only below (%)" uses. The addon never reads your
energy.

**What it can't see:**
- crits, so the extra combo point from Primal Fury
- Clearcasting procs
- whether you're behind the target
- target health

When you see a Clearcasting proc or you're in front of the mob, trust your eyes. Shred needs you
behind the target; Claw doesn't.

## How good is it

These numbers are from the simulator, not the game. The comparison is against a full-information
priority list built from the Oct 1 guides, which an addon couldn't run on Forever anyway. Following
StimCoach's suggestions:
- **Single-target boss:** within about 2% (−8 ± 4 DPS)
- **Boss with add waves:** slightly ahead (+9 ± 4)
- **Trash:** ahead (+22 ± 10)

It hasn't been tested in the real client yet. If something's off (an icon never changes, a spell
isn't recognized), the likely causes are a spell named differently in game or an API behaving
differently in combat.

## Regenerating

`Data.lua` is generated. Don't edit it.

```bash
uv run python scripts/gen_data.py --out runs/addon1 --specs feral --episodes 700
uv run python scripts/train_addon.py --data runs/addon1 --spec feral --out runs/addon1/addon_feral.pt
uv run python scripts/export_addon.py --spec feral --model runs/addon1/addon_feral.pt
```

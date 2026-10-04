# stim: handoff

Last updated 2026-10-04, session 2 (in progress).

## TL;DR

stim is a simulation-trained rotation brain for WoW, targeting WoW Forever first (Feral Druid is
the first spec you'll play). Instead of a hand-written priority list, a search-based teacher finds
the best ability by simulating ahead, and a small Jev-style decision model learns to make the same
call in about a millisecond.

**Done:** a class-agnostic melee combat engine where every class is a TOML data file. There are
three specs (Feral, Combat Rogue, Fury Warrior), three encounter types, baseline policies (random,
class-agnostic greedy, and a Hekili-style priority-list interpreter), and 21 passing tests.

**Session 2 (all uncommitted):**
- **The brain pipeline works end to end:** rollout teacher, entity-token features, Jev-style
  model, training, and evaluation. Students beat the baselines in most fights and beat the noisy
  teacher they learn from.
- **Expert iteration works:** round 1 fixed single-target Rogue.
- **Feral is rebuilt** to the Oct 1 beta kit.
- **The front end is now an in-game addon, StimCoach** (`addon/StimCoach`, Feral). It flashes the
  next 3 spells and plays about even with the full-information priority list in the sim, using only
  what Forever lets addons see. Energy comes from game-side gating. It's untested in the real
  client.
- **The practice trainer** (`stim.trainer`) works but is shelved.

**Next:** try StimCoach in game and fix what breaks (spell names, API behavior in combat). Then
Clearcasting gating, expert iteration for the addon, then more specs.

## Getting set up

The session-1 working copy lived in a temporary folder that will be deleted, so clone from GitHub:

```bash
gh repo clone jaylamping/stim ~/code/stim
cd ~/code/stim
uv sync
uv run pytest
uv run python scripts/baselines.py      # DPS of each baseline policy per spec and scenario
```

Python 3.13 is managed by uv. Dependencies are numpy and torch; torch isn't used yet, but the model
will need it.

## Why the design looks like this (research findings)

1. **WoW Forever blocks in-game decision addons.** Forever launches Nov 4, 2026 (beta through
   Oct 21). It runs on the mainline UI with Midnight's addon restrictions. In combat, health,
   auras, and threat come back as "secret values" that addons can display but not compute with.
   Auras can't be read at all in combat, and `CombatLogGetCurrentEventInfo` is unavailable. On the
   beta, even combo points are secret; a forum report says Blizzard plans to add them to the
   Personal Resource Display. Hekili ended with Midnight's prepatch. So stim runs outside the game.
   The beta's level cap is 30 as of Oct 3, so level-60 numbers come from datamined client data,
   not play. Tier 1 raids open Dec 9.
2. **Hard lines.** stim never reads the client outside the official addon API (no screen capture,
   no memory reading) and never sends input to the game. Every button press is the player's own.
   In the game it runs as an addon using only what the API exposes (decided in session 2, see
   "In-game addon").
3. **What is allowed: combat logs.** `/combatlog` writes to disk. Advanced Combat Logging includes
   unit positions and facing (worth confirming on a real Forever log). Chronicle already parses
   Forever beta logs (combat log V22), and foreverlogs.gg exists. A post-pull coach built on logs
   is in the same category as Warcraft Logs and WoWAnalyzer.
4. **Jev** (TypeSafe AI, early access since Sept 15, 2026) is proprietary. You send a block of
   state plus typed questions (Choice, Score, or Noul, which is yes/no), and it returns calibrated
   probabilities and a confidence. The vendor claims 70-500 ms per call. It has no vision or
   spatial ability of its own; it only knows what's in the state you pass. **Open Jev**
   (`kyegomez/open-jev`, Apache-2.0) is an untrained reconstruction. Its hash tokenizer isn't
   trainable, and it encodes numbers as hashed strings. We're borrowing its design ideas, not its
   code:
   - a bidirectional state encoder
   - learned query slots that cross-attend into the state
   - a Choice head that scores runtime-supplied options as `q · option`
   - an ordinal Score head, a Noul head, and a separate confidence head
   - the RLCD loss: soft-target NLL + Brier + consistency + evidential + ECE, with the rule
     "never train on hard labels"
5. **Forever Feral kit, as of Blizzard's Oct 1 beta changes** (re-checked Oct 4; replaces the
   session-1 list, which got Primal Bite wrong). "Client data" means numbers datamined from beta
   client 1.60.1.69893 (Sept 17) in MythicSim's engine docs, which predate the Oct 1 changes.
   Damage numbers are the level-60 rank before attack power scaling.

   | Ability | In Forever | Source |
   | --- | --- | --- |
   | Tiger's Fury, King of the Jungle | **Removed in the Oct 1 beta update.** Before that, TF was free with a 30 s cooldown and gave +15% physical damage for 6 s. | Blizzard dev notes via classicwowforever; forum thread |
   | Shifting Power | Feral talent: +40 energy in Cat Form for 55% of base mana, 16 s cooldown. Improved Shifting Power cuts it by 4/8 s. Guides say to cast it below 60 energy. | classicwowforever; Icy Veins; forum thread |
   | Claw | Cat builder with no positional requirement: 45 energy, 110% weapon damage plus a flat bonus, 1 CP. | client data; classicwowforever |
   | Shred | 60 energy, from behind: 155% weapon damage (Classic 225%) plus a flat bonus, 1 CP. Shredding Attacks takes 6 energy off per rank. | client data; classicwowforever |
   | Rake | 40 energy: 61 damage plus 34 per tick over 9 s, 1 CP. Guides rank it below Shred. | client data; Icy Veins |
   | Rip | 30 energy finisher: 15 + 25.5 per CP per tick, 6 ticks over 12 s. | client data |
   | Ferocious Bite | Unchanged from Classic (learned at 32). | client data |
   | Faerie Fire | The Feral talent version is gone, but Faerie Fire itself is free with a 6 s cooldown in animal forms and no longer resets the swing timer. | client data; Blizzard class deep dive via classicwowforever |
   | Primal Bite | Bear Mangle, renamed Sept 24. **Bear Form only**, not a cat builder. | classicwowforever; Icy Veins |
   | Feral Charge | Talent (10 Feral points). Bear: 5 rage, 15 s cooldown, roots and interrupts. Cat: leaps behind the target, 30 s cooldown, no root. | classicwowforever |
   | Berserk | 31-point talent: instant, 3 min cooldown, lasts 15 s, no cost, fear immunity. Its cat effect isn't published; the demo text only describes Bear Mangle hitting 3 targets with no cooldown. We assume Wrath's halved energy costs. | client data; forum; wowforeversim demo text |
   | Primal Fury | 50% / 100% chance of an extra combo point on non-periodic builder crits. | client data |
   | Omen of Clarity | Learned from a trainer now, not talented. | wowforeversim; Icy Veins |
   | Furor | Nerfed so shifting no longer farms energy. Powershifting is effectively gone. | Icy Veins; Blizzard notes |
   | Forms | Form attacks use the equipped weapon's DPS; the form sets the swing speed. Cat Form costs 55% of base mana. | classicwowforever; Icy Veins |
   | Thistle Tea in forms | Reported in session 1, not re-verified. | — |

   Blood Frenzy is unclear: the client data folds its combo-point proc into Primal Fury, but
   Icy Veins' Oct 1 build still takes a point in it.
6. **Forever rule changes that affect the engine.**
   - DoTs and HoTs can crit (Icy Veins). MythicSim rolls them with the crit chance snapshotted at
     application. Its evidence is tooltip wording: "non-periodic" qualifiers on Primal Fury and
     Nature's Grace, and the Warlock talent Pandemic in the Forever tree. Nothing found says
     refreshes keep remaining duration (modern "pandemic"), so refreshes stay vanilla-style. Our
     engine doesn't crit DoTs yet.
   - Rogue (client data): Rupture ticks 35 + 4.73 per CP (Classic 60 + 8), Instant Poison 76-100
     (Classic 112-148), Deadly Poison ticks 23 (Classic 34). Sinister Strike, Eviscerate, Slice and Dice,
     Blade Flurry, and Adrenaline Rush are unchanged. Poison proc chances stay 20% / 30%.
   - Warrior (client data): Bloodthirst 35% of AP + 48. Battle Shout was cut about 40% (rank 7:
     139 AP for 3 min), and Improved Battle Shout isn't baseline. Enrage is a flat 30% chance for
     2-10% physical damage. Deep Wounds is 20/40/60% of weapon damage over 12 s in 4 ticks.
     Flurry is 5-25% haste, and Unbridled Wrath is 12% per point. Slam has a 15 s cooldown and no
     longer resets the swing. Heroic Strike, Cleave, Execute, Whirlwind, Bloodrage, and Death Wish
     are unchanged.
7. **Level-60 reference points.** MythicSim's engine predates Oct 1, so these still include
   Tiger's Fury. Its Feral preset sims about 519 DPS (2 min single target, raid buffs), and the top
   community build about 780. Damage shares: white swings about 47% (37.5% plus 9.9% Windfury
   extra attacks), Shred 29%, Rip 17%, Ferocious Bite 3%. `valuation.WHITE_SHARE = 0.45` agrees.
8. **Where numbers come from.**
   - **MythicSim's engine** ([sage3648/mythicsim-forever-engine-go](https://github.com/sage3648/mythicsim-forever-engine-go),
     MIT, built on wowsims/classic) is the best source of datamined numbers:
     - `docs/beta-pass/<class>.md`: per-class client numbers
     - `docs/forever_rules.md`: rule changes
     - `docs/data-changes/`: per-build diffs
   - **classicwowforever.com**'s class guides cite the client build or Blizzard note behind each line
     and are the most current on the kit.
   - **Wowhead** has a Forever database (`wowhead.com/forever/...`).
   - **foreverlogs.gg** has public beta combat logs, only from leveling dungeons for now. Its API needs
     a free key, requires attribution, and forbids bulk redistribution. It's useful later for the
     coach's log format.
   - **wowforeversim.com** has BlizzCon-demo talent data (Sept 15), now superseded.

Sources:
- [Blizzard: WoW Forever](https://news.blizzard.com/en-us/article/24302093/carve-a-new-path-with-world-of-warcraft-forever)
- [Icy Veins: addons in Forever](https://www.icy-veins.com/wow-forever/news/addons-in-wow-forever-blizzard-devs-just-addressed-the-big-question/)
- [Forums: combo points are secret](https://us.forums.blizzard.com/en/wow/t/combo-points-are-secret-values/2352853)
- [Jev on Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- [open-jev](https://github.com/kyegomez/open-jev)
- [Chronicle Forever log support](https://github.com/Emyrk/chronicle/pull/694)
- [Icy Veins Forever druid overview](https://www.icy-veins.com/wow-forever/druid-class-overview) (Sept 20)
- [Icy Veins Forever Feral guide](https://www.icy-veins.com/wow-forever/feral-druid-melee-dps-and-tank-pve-guide) (Oct 1, level 30)
- [classicwowforever Feral guide](https://classicwowforever.com/class-guides/druid/feral/) (Oct 3, client 1.60.1.70009 plus Blizzard's Oct 1 notes)
- [Forums: "Feral changes are BAD"](https://us.forums.blizzard.com/en/wow/t/feral-changes-are-bad/2369080) (Oct 1-2, Tiger's Fury removal and Shifting Power)
- [MythicSim Feral build with ability breakdown](https://mythicsim.com/wow-forever/builds/bad53cd6-bbf8-418d-9a12-8e88fda65e25?scenario=st-120-v1)
- [MythicSim engine docs](https://github.com/sage3648/mythicsim-forever-engine-go/tree/master/docs)
- [Forever Logs statistics](https://foreverlogs.gg/statistics?phase=1&location=All+Dungeons) and [API terms](https://foreverlogs.gg/docs/api)
- [wowforeversim Feral sim](https://wowforeversim.com/sim/feral_druid) (BlizzCon demo data)

## Decisions so far

- The name is **stim** ("sim" with a T, and a stim is a boost). The repo is private.
- **The engine is class-agnostic.** Every class/spec is a swappable data file, and nothing in
  `src/stim` knows which class it's simulating. Melee classes come first; casters and healers
  come later.
- **The front end is an in-game addon** (decided in session 2). It shows a WeakAuras-style strip
  around the character with the next few recommended spells, computed in Lua from what the addon
  API allows. The practice trainer (`stim.trainer`) works but is shelved; the post-pull coach comes
  later.
- The learned model makes the decisions. Priority lists (`[[apl]]` in specs) are baselines to beat.

## What exists

| Path | What it is |
| --- | --- |
| `specs/*.toml` | Feral (built from the session-1 kit reports, now out of date; see "Spec corrections needed"), Combat Rogue and Fury Warrior (classic-era kits, there to prove genericity). All numbers are placeholders. |
| `src/stim/spec.py` | Schema, loader with strict validation (unknown fields raise `SpecError`), and `Spec.randomized()` for domain randomization |
| `src/stim/sim.py` | The engine: `Rules` (spec compiled to index lookups), `Sim`, `Enemy`, `EventPlan` |
| `src/stim/scenarios.py` | `boss`, `boss_adds`, and `trash` encounter generators |
| `src/stim/valuation.py` | Expected-value math (expected strike damage, DoT gain vs. clipping loss, buff/debuff value). Used by the greedy policy and planned as model features. |
| `src/stim/policies.py` | `RandomPolicy`, `GreedyPolicy` (class-agnostic), `AplPolicy` (interprets `[[apl]]`), `apl_namespace`, `EpsilonPolicy` (behavior noise), `best_baseline` (picks greedy or APL per spec by measured DPS) |
| `src/stim/teacher.py` | Rollout teacher: `Teacher.evaluate` → `Label` (raw Q matrix, soft target, regret, stakes, survival); `rollouts` (lockstep, batched-policy ready); `state_value` (terminal value); `summarize`; `TeacherPolicy` |
| `src/stim/data.py` | Episode generation with a behavior policy plus teacher labels; `snapshot`/`restore` decision states (no future in them); pickled shards |
| `src/stim/features.py` | Entity tokens: player, resources, auras, abilities, enemies, options (`tokens`, `VERSION`) |
| `src/stim/model.py` | `StimNet`: Jev-style encoder, query-slot readout, Choice/Noul/Score/Confidence heads (about 600k parameters); `Batch` padding |
| `src/stim/dataset.py` | Records → `Example`s (tokens + targets), cached per run; split by episode |
| `src/stim/student.py` | `StudentPolicy` (single and batched decisions), checkpoint save/load |
| `tests/test_engine.py` | Every spec × scenario × policy runs; APL beats random; mechanics tests (energy ticks, CPs on target, finisher scaling, positionals, gap closer, rage, Heroic Strike, clone independence, determinism, schema errors) |
| `tests/test_teacher.py` | Aligned luck across branches, expected damage vs. actual, event memory, randomized-spec rules, terminal-value pieces, soft targets, exact common random numbers, no access to the real future, batched rollout policies |
| `tests/test_model.py` | Tokens finite and well formed for every spec and scenario, rotation invariance, masking, batched = single decisions, checkpoint round trip |
| `scripts/baselines.py` | Baseline DPS table |
| `scripts/calibrate_values.py` | Measures each resource's marginal value under a policy; `--write` updates `resource_value` |
| `scripts/gen_data.py` | Parallel labeled-data generation into `runs/<name>/` |
| `scripts/train.py` | Trains the model; reports held-out regret against the teacher next to the baselines' |
| `scripts/evaluate.py` | Full-fight DPS of any policies on paired seeds (`greedy`, `apl`, `student=<ckpt>`, `addon=<ckpt>`, `teacher:<rollout>`), optional spec randomization, decision latency |
| `src/stim/addon/` | The in-game addon's Python side: `tracker.py` (what the addon can know, energy bands), `policy.py` (small MLP, `AddonPolicy` follower, examples), `export.py` (Data.lua and .toc) |
| `scripts/train_addon.py`, `scripts/export_addon.py` | Train the addon policy; write the addon's data |
| `addon/StimCoach/` | The addon: `Tracker.lua`, `Policy.lua`, `Core.lua`, generated `Data.lua`, `.toc`, README |
| `src/stim/trainer/` | Practice trainer (shelved): `session.py` (game logic, review), `server.py` (WebSocket), `static/index.html`; run with `python -m stim.trainer` |
| `tests/test_addon.py`, `tests/test_trainer.py` | Lua/Python parity and mocked-WoW addon tests; trainer session tests |

### Engine semantics

- **Options.** A decision point offers options as `(ability index, enemy id)` pairs. `POOL` (-1)
  means wait until the next swing or resource tick, at most 1 s. Targeted abilities produce one
  option per valid enemy. AoE-around-player and next-swing abilities are untargeted.
- **Decision timing.** Decisions happen when the GCD is ready. Off-GCD abilities resolve at the
  decision point without advancing time; there's no weaving inside the GCD. Next-swing abilities
  (Heroic Strike) queue and resolve on the next main-hand swing; that swing generates no rage.
- **Finishers** consume their whole resource. Target-bound resources (combo points) only count on
  the enemy that holds them. Building on a different enemy resets them, which matches vanilla
  and Forever, where combo points live on the target.
- **Hit resolution.** Abilities with damage roll miss and dodge. If the primary target isn't hit,
  the DoT, debuff, and gains don't apply. White swings use one roll (miss, dodge, glancing), then
  a crit roll on non-glancing hits.
- **Positioning.** The tank stands at a point. Enemies walk to the tank and face it. The player
  auto-runs to the spot 2.5 yd behind the current target, so the model doesn't control movement,
  only gap closers.
- **Disruptions.** Knockbacks (12-18 yd, no actions for 0.6 s), boss turns (break "behind"), tank
  repositions, and add waves.
- **No future knowledge.** `Sim.resample_future(rng)` replaces upcoming scripted events with a
  fresh draw. The teacher must use it so it never sees events the player couldn't know about.
  Each event kind recurs at uniform random intervals. `Sim.last_event` records when each kind last
  happened, and the draw is conditioned on the time elapsed since then, which a player can see.
  Without that, the teacher always thought the next knockback was 18-35 s away.
- **Randomness.** Combat rolls come from five streams: main hand, off hand, ability uses, splash
  targets, and nested procs. Every attack draws a fixed block (hit roll, crit roll, one roll per
  proc), so clones reseeded with the same seed get the same luck on the n-th swing or n-th ability
  use whatever happened in between. `clone(seed)` gives fresh rolls; `clone()` replays the
  original's.
- **Expected damage.** `Sim.ev_damage` adds each strike's mean over its hit and crit rolls, given
  the state before the roll. It tracks `damage` to within ±0.5% over 300 fights per spec and
  scenario, with far less noise; the teacher scores with it.
- **Time to die.** `Enemy.time_to_die` is HP divided by an observed health-loss EMA (τ = 3 s),
  which is what a player could estimate.
- **Damage accounting** excludes overkill.
- **Speed.** About 35k-200k decisions per second in pure Python, depending on spec and policy.

### Baselines (30 fights each, mean DPS ± standard error)

Rerun in session 2 after the random-stream change and after Rogue and Warrior `reference_dps`
were recalibrated (800 → 450 and 900 → 400). `reference_dps` sets boss HP, so their fights are
shorter now.

| Spec | Scenario | random | greedy | APL |
| --- | --- | --- | --- | --- |
| Feral | boss | 319 ± 4 | 378 ± 5 | **437 ± 7** |
| Feral | boss_adds | 297 ± 3 | 369 ± 4 | **417 ± 5** |
| Feral | trash | 360 ± 8 | 487 ± 11 | **548 ± 8** |
| Rogue | boss | 388 ± 7 | 413 ± 6 | **451 ± 7** |
| Rogue | boss_adds | 385 ± 4 | 412 ± 5 | **449 ± 6** |
| Rogue | trash | 733 ± 28 | 913 ± 33 | 894 ± 30 |
| Warrior | boss | 269 ± 9 | 345 ± 9 | **394 ± 9** |
| Warrior | boss_adds | 296 ± 7 | 379 ± 8 | **424 ± 8** |
| Warrior | trash | 383 ± 15 | 508 ± 17 | 558 ± 17 |

Feral rows are the rebuilt Oct 1 kit (40 fights). Before the rebuild, greedy beat the Feral APL;
with the client numbers and the guides' Shifting Power threshold (below 60 energy), the APL leads.

## Spec schema reference

```toml
[class]       name, spec, notes
[combat]      gcd, crit_chance, crit_multiplier, melee_range, run_speed, miss_chance,
              dodge_chance, glancing_chance, glancing_multiplier, dual_wield_miss
[weapons]     main_hand = { damage, speed }, off_hand = { damage, speed }   # off_hand optional

[resources.<id>]   kind = "energy" | "rage" | "mana" | "secondary", max, start,
                   regen_per_sec, tick = { amount, interval }, per_white_damage, on_target

[buffs.<id>] / [debuffs.<id>]
    name, duration, duration_by_points, max_stacks, charges,
    consumed_by = "cost" | "swing" | "ability:<id>" | <school>   (school: debuffs only)
    haste, damage_mult, crit_bonus,
    cost_mult = 0.5 | { <resource> = mult },  regen_mult = { <resource> = mult },
    cleave_targets, cleave_radius, damage_taken = { <school> = bonus per stack }

[[procs]]     id, trigger, chance | ppm, buff, debuff, gain = {..}, extra_attacks,
              damage = {..}, dot = {..}, requires_buff
              trigger: white_hit | yellow_hit | hit | crit | white_crit | yellow_crit |
                       builder_crit | dodge | ability:<id>

[[abilities]] id, name, key, cost = {..}, gain = {..}, set = {..}, cooldown, cooldown_group,
              gcd, off_gcd, cast_time, finisher = "<resource>", buff, debuff, consumes_buff,
              next_swing, gap_closer, range = [min, max],
              damage = { weapon, flat, by_points, school, can_crit, crit_bonus,
                         aoe_radius, max_targets, extra_targets },
              dot = { ticks, interval, per_tick, per_tick_per_point, school },
              extra = { resource, max, damage_per },
              requires = { behind, target_health_below, target_health_above, buff, no_buff }

[teacher]     reference_dps, resource_value = { <resource> = damage per unit }

[[apl]]       use = "<ability>", if = "<expression>"
              expression variables: <resource>, <resource>_max, <resource>_deficit,
              buff.<id>.up/remains/stacks, debuff.<id>.up/remains/stacks (current target),
              dot.<ability>.up/remains/ticking, cooldown.<ability>.ready/remains,
              target.health_pct/time_to_die/distance, behind, in_melee,
              enemies (within 8 yd), active_enemies, time
```

The schools are physical, bleed, nature, fire, frost, arcane, holy, and shadow. Bleeds are their
own school, so armor-style `damage_taken = { physical = .. }` debuffs don't boost them.

To add a class, write a spec file and run `uv run pytest`; the parametrized tests pick it up
automatically. If it needs a mechanic the engine lacks, add it as a generic primitive in
`spec.py` (schema plus validation) and `sim.py`, with a test. Never special-case a class.

## Spec corrections needed

From the research above (items 5-8). None of this blocks the teacher or the model: training
randomizes spec numbers, and the model reads them as features.

- **`feral.toml`: done in session 2.**
  - Kit: Claw replaces Primal Bite; Shifting Power is +40 energy for 684 mana on a 16 s cooldown;
    Feral Charge has a 30 s cooldown; Faerie Fire is kept.
  - Numbers: client values for Shred, Claw, Rake and Rip, plus a realistic hit table (3% miss,
    1.5% dodge, 40% glancing at 70%).
  - Calibration: `reference_dps` 450 and measured `resource_value`.
  - New baselines: APL 437 boss / 417 boss_adds / 548 trash; greedy 378 / 369 / 487.
  - Still assumed: Berserk's cat effect (halved energy costs), Thistle Tea in forms, Shifting Power
    being on the GCD, and talents (Ferocity 5/5, Shredding Attacks 2/2, Primal Fury 2/2). The 2/2
    Improved Shifting Power (8 s cooldown) isn't modeled.
  - Students trained before the rebuild (`runs/gen0`, `runs/gen1`) are stale for Feral.
- **Rogue and Warrior numbers:** the client values in item 6.
- **Attack power:** the engine has no attack power, so AP-scaled parts (Bloodthirst's 35% of AP,
  Rip and Rake AP shares) need a reference AP folded into flat numbers.
- **Engine primitive Forever needs** (generic, with a test): DoT crits, a `dot.can_crit` (with a
  `[combat]` default) that rolls each tick at the crit chance snapshotted on application. Periodic
  crits don't trigger procs; the tooltips say "non-periodic".
- **Exchange rates:** recalibrate `[teacher] resource_value` from measured marginal values (see
  "Teacher: status and findings").

## Teacher: status and findings (session 2)

`src/stim/teacher.py` implements the design below, and `scripts/teacher_eval.py` plays full fights
with it against the baselines on identical fight seeds. Uncommitted, no tests yet. Details:

- **Rollouts** step all options × samples in lockstep (`rollouts()`), so a batched student can
  drive them later.
- **Terminal value** (`state_value`) adds:
  - banked resources at the spec's rates, scaled down if the fight ends within 3 s
  - DoT ticks that land before the target's projected death
  - remaining aura time at min(contribution rate, upkeep rate)
  - uses of cooldowns longer than the horizon that are still possible before the fight ends
- **Fight end:** if the fight ends inside the horizon, the time saved is worth reference DPS.
- **Soft targets** use a per-option temperature τ = √(SE² + floor²), where SE is the paired
  difference's standard error and the floor is 0.1 s of reference DPS.

What we learned:
1. **Variance reduction matters most.** Aligned random streams plus expected-damage scoring cut
   the per-sample noise of option differences 2-3.5× (4-12× in variance).
2. **Terminal-value traps (all fixed):**
   - Valuing an aura's remaining time at full DPS made the teacher refresh maintained auras early
     (23 Battle Shouts vs. the APL's 6). The fix values it at upkeep: the cheapest
     resources-plus-points per second to keep it up. GCD time counts as free because melee
     rotations are resource-bound.
   - Unused long cooldowns were worth 0, so it burned them instantly (Thistle Tea instead of
     Shred). The fix counts the uses left before the fight ends.
   - Memoryless event resampling (fixed in the engine, see above).
3. **Noise and the winner's curse.** At K = 8 the teacher overrides the Warrior APL on 14% of
   decisions with a median significance of z = 0.7, so it mostly chases noise. K = 16-32 fixes
   most of it. Longer horizons make it worse because they're noisier. Data generation should use
   K ≥ 16.
4. **Results:** 16 boss fights, paired against the APL on the same seeds.

   | Spec | Teacher | Δ DPS vs. APL |
   | --- | --- | --- |
   | Feral | greedy rollouts, K = 32 | +64 ± 9 (greedy itself +46) |
   | Feral | APL rollouts, K = 32 | +41 ± 9 |
   | Warrior | APL rollouts, K = 16 or 32 | −6 ± 5 (ties the APL) |
   | Warrior | greedy rollouts | about +14 over greedy |

   One-step lookahead is a modest improvement; expert iteration is what should compound it.
5. **The spec exchange rates are off by up to 50%.** Measured as marginal damage over 30 s from
   paired rollouts with and without extra resource, on boss fights:

   | Resource | Measured | Spec |
   | --- | --- | --- |
   | Warrior rage | about 23 (APL and greedy) | 15 |
   | Feral and Rogue energy | about 15 (APL) | 12 |
   | Feral mana | 0.8 | 1.5 |
   | Combo points | swings with the policy: Feral APL 86 ± 40, Rogue APL 162 ± 25, greedy 38-45 | 140 / 120 |

   This likely biases close calls. A calibration script should set `resource_value` per spec
   before data generation.

   `scripts/calibrate_values.py --write` has since set `resource_value` to measured values under
   each spec's rollout policy: greedy for Feral (energy 8.7, CP 66, mana 0.77, a noisy fixed point
   because greedy reads these values itself), APL for Rogue (energy 18, CP 230) and Warrior
   (rage 22). Effect on the teacher, paired over 24 fights:

   | Spec | Boss | Trash |
   | --- | --- | --- |
   | Feral | +8 ± 5 | +26 ± 12 |
   | Rogue | −5 ± 4 | −6 ± 7 |
   | Warrior | +3 ± 4 | +3 ± 5 |

   **Caveat:** the model also reads `resource_value` as a feature, so changing it shifts a
   trained student's inputs. Decouple the teacher's exchange rates from the feature-facing ones
   before recalibrating per iteration.

## Student: first results (session 2)

- **gen0 data:** 62,573 labeled states from 3,150 episodes (350 per spec and scenario), K = 16,
  randomized specs, behavior greedy or APL plus 15% random. Generated in 130 s on 16 workers.
- **Training:** 30 epochs on MPS in about 4.5 min. Best held-out regret is 0.085 s of reference
  DPS per decision (APL 0.161, greedy 0.259, random 0.529), with confidence ECE about 0.01.
- **Full fights:** 30 held-out seeds per cell, base specs. Δ is paired against the best baseline
  in that row.

  | Spec | Scenario | Student Δ | Teacher (K = 16, auto rollouts) Δ |
  | --- | --- | --- | --- |
  | Feral | boss | **+54 ± 5** | +34 ± 7 |
  | Feral | boss_adds | **+64 ± 5** | +53 ± 5 |
  | Feral | trash | +25 ± 12 | +19 ± 12 |
  | Rogue | boss | −9 ± 3 | −6 ± 3 |
  | Rogue | boss_adds | −3 ± 3 | −7 ± 3 |
  | Rogue | trash | +88 ± 20 | **+123 ± 22** |
  | Warrior | boss | +4 ± 3 | −8 ± 3 |
  | Warrior | boss_adds | **+13 ± 3** | −4 ± 4 |
  | Warrior | trash | +22 ± 10 | +16 ± 9 |

  The student beats the noisy teacher it learned from in most cells, because it averages the
  teacher's per-state noise across similar states. Decision latency: p50 0.95 ms, p99 1.67 ms
  (CPU, one thread). The weak spot is single-target Rogue, where the APL is hard to beat and the
  teacher's APL rollouts don't improve on it.
- **Expert iteration cost:** rollouts driven by the student cost 12-27× more than APL rollouts
  (about 0.3-1.4 s per labeled state), mostly the forward pass (13.5 ms per lockstep step of 128
  sims on one thread).
- **Expert iteration round 1** (`runs/gen1`): 30,925 states, the teacher rolling out with
  student0, behavior split across student, greedy and APL. Student1 trained on gen0 + gen1 and
  overfits after about 16 epochs. Full fights, 40 seeds, Δ DPS vs. the APL:

  | Spec | Scenario | Student0 | Student1 |
  | --- | --- | --- | --- |
  | Rogue | boss | −6 ± 2 | **+8 ± 3** |
  | Rogue | boss_adds | −2 ± 2 | **+6 ± 2** |
  | Rogue | trash | +91 ± 16 | +68 ± 16 |
  | Warrior | boss | +4 ± 3 | +4 ± 3 |
  | Warrior | boss_adds | +13 ± 3 | **+23 ± 4** |
  | Warrior | trash | +15 ± 8 | +22 ± 8 |

  Single-target Rogue now beats its strong APL. Feral wasn't compared: its kit changed under both
  students.

## In-game addon (session 2)

**Goal:** a barebones addon with a WeakAuras-style strip around your character that flashes the next
few spells, updating as you play. Every press is yours.

**What an addon can use in combat on Forever**, per WeakAuras Forever's tested list
([CurseForge](https://www.curseforge.com/wow/addons/weakauras-forever), client 1.60.1):
- Your own successful casts (`UNIT_SPELLCAST_SUCCEEDED` for the player) and `PLAYER_TARGET_DIED`.
- Range checks and the main-hand swing timer.
- Out of combat, everything (energy, mana, talents).

The game draws but hides from code: energy and mana values, combo points, buffs, cooldown values,
and target health. The combat log is off-limits to addons.

**Approach:**
1. **A tracker** (`stim.addon.tracker`, mirrored in Lua) rebuilds what it can from your own casts
   plus the spec's rules:
   - cooldowns and the GCD
   - your Rip, Rake, and Faerie Fire timers on the current target, and your Berserk window
   - estimated energy: pull value, average regen, costs, Shifting Power and Thistle Tea
   - builders since your last finisher on this target
   - estimated mana
   - range to the target

   Unknown to it: crits (so Primal Fury points), Clearcasting procs, target health, and whether
   you're behind.
2. **A policy trained on tracker features only,** with soft labels from the full-information
   teacher. It learns the best call under exactly that uncertainty. It is a small MLP, a few
   thousand weights, so it runs in Lua every frame.
3. **"Next X spells":** the addon rolls its tracker forward along its own recommendations, like
   Hekili's queue.
4. **Export:** a script writes the addon (tracker, weights, icon strip) for a spec. Lua is tested
   against Python test vectors with an embedded Lua runtime, since we can't run the game here.

**Status: built, Feral only, untested in the real client.** `addon/StimCoach/` (see its README) has:
- `Tracker.lua` and `Policy.lua`, which mirror the Python
- `Core.lua`: events, display, `/stim`
- generated `Data.lua` and `.toc`, written by `scripts/export_addon.py`

`tests/test_addon.py` replays fights through the Python and Lua trackers and checks that features,
masks, scores and the 3-spell queue agree. It also loads the whole addon against a mocked WoW API and
checks the energy gating shows the right band. A refresh (5 bands × a 3-spell queue, 15 policy
evaluations) takes 1.4 ms in Lua 5.1.

**The key finding: knowing energy is what matters, and the game can gate on it for us.** Retraining
the addon MLP with one oracle input at a time (boss fights, paired against the full-information APL):

| Addon knows | Boss Δ DPS |
| --- | --- |
| tracker only | −50 ± 6 |
| + true energy | −11 ± 6 |
| + Clearcasting | −29 ± 5 |
| + true combo points | −43 ± 5 |
| + behind or target health | no gain |
| + everything | +10 ± 5 |

The addon can't read energy, but `UnitPowerPercent(unit, powerType, false, curve)` evaluates a step
curve on the real value and the result can be a frame's alpha. That's what WeakAuras Forever's "Show
only below (%)" does (WeakAuras-Forever `Prototypes.lua`, `ThresholdAlpha`). So StimCoach
precomputes a queue for each of 5 energy bands, and the game shows the right one.

Policy trained with banded energy (`runs/addon1`: 42k Feral states, 8,843 weights), full fights vs.
the APL:

| Scenario | Δ DPS |
| --- | --- |
| boss | −8 ± 4 |
| boss_adds | +9 ± 4 |
| trash | +22 ± 10 |

Decisions take 0.09 ms in Python. Five uniform bands did as well as eight bands cut at the cost
thresholds, and gating combo points added nothing measurable. Gating Clearcasting would add about +8
on boss, but needs a native aura-presence gate (WeakAuras Forever's "Aura (Modern)" containers), so
it's left for v2.

**Simulating a player who follows the addon** (`AddonPolicy`):
- Press the top suggestion when the game allows it.
- Wait for the button to light up if it's unaffordable.
- Take the next suggestion if it's impossible for another reason (not behind, out of range).

**Ideas for "getting clever" next:**
- Clearcasting gating.
- Expert iteration for the addon's teacher: the addon student as the rollout policy, with its
  tracker cloned alongside the sim.
- `UI_ERROR_MESSAGE` ("must be behind") to learn positional state.
- A shapeshift-form check, so the addon hides in bear form.

## Next steps, in order

1. **Try StimCoach in game** (Feral, beta). Likely fixes:
   - spell names that differ in game
   - APIs that behave differently in combat (range checks, `UnitPowerPercent` gating)
   - whether the strip should hide in bear form
2. **Clearcasting gating** for the addon (about +8 DPS on boss in the sim). It needs a native
   aura-presence gate; WeakAuras Forever's "Aura (Modern)" containers show how.
3. **Expert iteration for the addon:** a teacher rolling out with the addon student, with its tracker
   cloned alongside the sim, and more Feral data.
4. **Engine:** DoT crits (Forever rule), attack power, and the remaining Feral assumptions.
5. **More specs in the addon.** Rogue is straightforward (energy gating). Warrior gates on rage,
   which the game can also gate, even though the addon can't estimate rage from white damage.
   Then Ret Paladin and Enhancement Shaman.
6. **Post-pull coach** from combat logs, once level-60 logs exist. Then the shelved practice trainer.

## Design notes (as built in session 2)


### Teacher (`src/stim/teacher.py`)

- For a decision state with legal options A, run K samples (K ≥ 16; K = 8 is too noisy, see the
  findings). Each sample k gets its own future-event draw and RNG seed, and that draw is shared by
  every option (common random numbers).
- For each option a:
  1. clone the sim with sample k's seed
  2. give it sample k's event draw
  3. `step(a)`
  4. follow the rollout policy until t0 + H (H = 20 s)
- Score each rollout: Q_k(a) = expected damage gained + the terminal value `state_value`
  (resources, DoT ticks, auras at upkeep, long-cooldown uses; see the findings). If the fight ends
  early, add the time saved at reference DPS.
- **Soft targets:** p(a) ∝ exp(−gap(a) / τ(a)), with τ(a) = √(SE(a)² + floor²). SE(a) is the
  standard error of option a's paired difference to the best option. The floor is 0.1 s of
  reference DPS. Never use hard labels. Store the raw Q matrix so targets can be recomputed.
- **Extra labels:**
  - **Stakes** (Score, 4 levels): mean regret over legal options, in seconds of reference DPS,
    with cutoffs at 0.25, 1, and 2.5.
  - **Target survives** (Noul): P(current target is alive at t0 + 12 s), from the rollouts.
- **Rollout policy:**
  - Iteration 0: the better of greedy and APL for that spec.
  - Iteration 1 onward: the previous student, with inference batched across all rollouts stepping
    in lockstep.
- **Visiting states:** play episodes with a mixed behavior policy: student, APL, or greedy, plus
  15% random actions, and the teacher's best when a state is labeled. Off-policy states matter
  because trainer users will make mistakes.
- Use `Spec.randomized(rng)` per episode so the model learns to read the numbers. That's how tuning
  hotfixes avoid a retrain.
- **Cost (measured):** a 20 s rollout costs about 0.3 ms. A single-target state takes about 17 ms at
  K = 8 and scales with K and the option count, so trash pulls with several targets cost a few
  times more. With 16 worker processes (the M5 Max has 18 cores), 40k states at K = 16 take a few
  minutes.

### Features (`src/stim/features.py`)

These are generic entity tokens, padded per type. Sizes are not tuned.

- **player:** GCD, swing timers and speeds, haste, crit, knocked back, time, enemy counts (total,
  in melee, within 8 yd)
- **resources:** kind one-hot, value/max, max, regen rate, time to next tick, target-bound, bound
  to the current target
- **active buffs:** remaining/duration, stacks, charges, effect magnitudes
- **every ability, legal or not:** a static descriptor built from the TOML (costs by resource kind,
  cooldown, GCD, cast time, damage coefficients, DoT, AoE radius, flags), plus cooldown remaining,
  whether it's affordable, and whether it's legal
- **enemies:** relative x/y, distance, in melee, behind, HP fraction, log HP, loss rate, time to
  die, own DoT and debuff remaining, whether it's the target, whether it holds combo points
- **options:** a reference to the ability token and the target token, plus `expected_strike`,
  `expected_dot` gain and clip loss, targets hit, effective cost, resources after, whether it
  would reset combo points, points consumed, and buff/debuff coverage gained

There are no ability-ID embeddings by default, so abilities stay swappable and a model trained
across specs can try new ones zero-shot. Augment training data by rotating x/y about the player.

### Model (`src/stim/model.py`): Jev-style, about 0.5M parameters, target under 1 ms on CPU

- **Encoder:** d = 96, 4 heads, a 3-layer bidirectional transformer over all tokens with type
  embeddings. Option tokens also add the input embeddings of their ability and target tokens.
- **Readout:** 4 learned query slots per question and 2 cross-attention readout blocks.
- **Heads:**
  - Choice: `q · option / √d / learned temperature`, masked to legal options
  - Noul: will the target survive?
  - Score: ordinal stakes
  - Confidence: a sigmoid, trained to predict the probability mass on the teacher's best option

### Loss: RLCD-style

- Choice: soft-target NLL + 0.5 × Brier + 0.2 × evidential term. The evidential term compares
  confidence to agreement with the teacher, with stop-gradient.
- Noul: BCE + Brier.
- Score: ordinal NLL.
- ECE is tracked as a metric only.

### Data generation and training

Use multiprocess workers writing `.npz` shards under `runs/` (gitignored). Train with AdamW and a
cosine schedule. Split validation by episode. Do 2-3 rounds of expert iteration.

### Evaluation

- Full-fight DPS on held-out seeds for random, greedy, APL, teacher, and student.
- Regret against the teacher on held-out labeled states. This is more meaningful than top-1
  agreement, since near-ties don't matter.
- Calibration: ECE and reliability.
- Latency: p50 and p99.
- Robustness: does the model adapt to perturbed tuning without retraining?

### Practice trainer (built in session 2, shelved)

- A browser front end plus a local Python websocket backend that runs the same `Sim` on a real
  (scalable) clock.
- **Screen:**
  - a 2D map with positions, facing, and the behind arc
  - resource bars
  - an ability bar using the spec's `key` fields
- **Coaching:**
  - Highlight the model's top choice with confidence and stakes. It can re-decide every frame.
  - Record what you press.
  - After the pull, list the decisions with the biggest regret.

### Post-pull coach

Parse `WoWCombatLog.txt` (advanced logging), rebuild state at each GCD, and grade your choices
against the model. This needs a real Forever log to build against.

### More specs, then casters and healers

- **Ret Paladin** needs seals as buffs with on-swing procs, and Judgement consuming a seal.
- **Enhancement Shaman** needs shared shock cooldowns (`cooldown_group` exists), Stormstrike
  charges (supported), and cast times (basic support exists).
- **Casters and healers** come after that.

## Known simplifications

- **Placeholders everywhere.** Every number in every spec is a placeholder; real values are listed
  in the research section and "Spec corrections needed". Rogue and Warrior `reference_dps` were
  recalibrated to the sim's output (450 / 400) in session 2. It sets boss HP and the valuation
  scale.
- **No stances or forms.** Stance-gated abilities like Overpower aren't modeled.
- **Thin hit table.** No parry, block, or resource refunds on miss or dodge.
- **Forever DoT crits not modeled yet.** In Forever, DoTs can crit; here they never do. Refreshing
  a DoT resets it (vanilla-style), which matches what's known about Forever.
- **Swing timing.** Haste applies when the next swing is scheduled; there's no mid-swing rescale.
  Extra attacks don't reset the swing timer.
- **Encounter simplifications.** Party damage per enemy is constant. Boss death ends boss fights
  even if adds are alive. There's no mana five-second rule.
- **Crude valuation.** `valuation.py` constants (`SCHOOL_SHARE`, `WHITE_SHARE`) are rough. They
  only affect the greedy policy and model features.
- **APL eval.** APL `if` strings are `eval`'d with no builtins. That's fine for local spec files,
  but don't load untrusted specs.

## Open questions for you

- Do you have Forever beta access? With it we can read real tooltip numbers and grab a combat log
  for the coach. The beta cap is 30, so level-60 numbers will still come from client data.
- When you try StimCoach in game: does each cast register (the strip changes as you press), does the
  energy gating switch strips as your energy moves, and do any spells have different names?
- Which talents do you take? `feral.toml` assumes Ferocity 5/5, Shredding Attacks 2/2, Primal Fury
  2/2 and Berserk, and not Improved Shifting Power.
- What are your actual keybinds? The spec `key` fields are placeholders.
- Should talents be separate spec files (`feral_bleed.toml`, ...) or a `[talents]` section that
  patches the base spec?

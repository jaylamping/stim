# stim: handoff

Last updated 2026-10-04, end of session 1.

## TL;DR

stim is a simulation-trained rotation brain for WoW, targeting WoW Forever first (Feral Druid is
the first spec you'll play). Instead of a hand-written priority list, a search-based teacher finds
the best ability by simulating ahead, and a small Jev-style decision model learns to make the same
call in about a millisecond.

**Done:** a class-agnostic melee combat engine where every class is a TOML data file. There are
three specs (Feral, Combat Rogue, Fury Warrior), three encounter types, baseline policies (random,
class-agnostic greedy, and a Hekili-style priority-list interpreter), and 21 passing tests.

**Next:** the teacher (rollout search), then features, the Jev-style model, training, evaluation,
and the practice trainer UI. The design for each is written up below so the next session doesn't
have to re-derive it.

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
   Oct 21). It runs on the mainline UI with Midnight's addon restrictions. In combat, health, auras,
   and threat come back as "secret values" that addons can display but not compute with. Auras
   can't be read at all in combat, and `CombatLogGetCurrentEventInfo` is unavailable. On the beta,
   even combo points are secret; a forum report says Blizzard plans to add them to the Personal
   Resource Display. Hekili ended with Midnight's prepatch. So stim runs outside the game.
2. **Hard lines.** stim never reads the live client (screen capture, memory) to feed decisions and
   never sends input to the game. Either one is a third-party tool doing what Blizzard restricted,
   which is the route that gets accounts banned.
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
5. **Forever Feral kit** (beta reports, Oct 1-4, 2026):
   - Tiger's Fury and King of the Jungle are removed.
   - Mangle is renamed **Primal Bite** and loses its debuff.
   - Powershifting becomes a talent, **Shifting Power**. Guides say to cast it below 60 energy.
   - **Berserk** is the new 31-point talent (15 s offensive window).
   - Cat **Feral Charge** leaps behind the target.
   - Primal Fury adds an extra combo point on builder crits.
   - Consumables such as Thistle Tea work in forms.

Sources:
- [Blizzard: WoW Forever](https://news.blizzard.com/en-us/article/24302093/carve-a-new-path-with-world-of-warcraft-forever)
- [Icy Veins: addons in Forever](https://www.icy-veins.com/wow-forever/news/addons-in-wow-forever-blizzard-devs-just-addressed-the-big-question/)
- [Forums: combo points are secret](https://us.forums.blizzard.com/en/wow/t/combo-points-are-secret-values/2352853)
- [Jev on Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- [open-jev](https://github.com/kyegomez/open-jev)
- [Chronicle Forever log support](https://github.com/Emyrk/chronicle/pull/694)
- [Icy Veins Forever druid guide](https://www.icy-veins.com/wow-forever/druid-class-overview)

## Decisions so far

- The name is **stim** ("sim" with a T, and a stim is a boost). The repo is private.
- **The engine is class-agnostic.** Every class/spec is a swappable data file, and nothing in
  `src/stim` knows which class it's simulating. Melee classes come first; casters and healers
  come later.
- The first front end is the **practice trainer**. The post-pull coach comes after that.
- The learned model makes the decisions. Priority lists (`[[apl]]` in specs) are baselines to beat.

## What exists

| Path | What it is |
| --- | --- |
| `specs/*.toml` | Feral (Forever beta kit), Combat Rogue and Fury Warrior (classic-era kits, there to prove genericity). All numbers are placeholders. |
| `src/stim/spec.py` | Schema, loader with strict validation (unknown fields raise `SpecError`), and `Spec.randomized()` for domain randomization |
| `src/stim/sim.py` | The engine: `Rules` (spec compiled to index lookups), `Sim`, `Enemy`, `EventPlan` |
| `src/stim/scenarios.py` | `boss`, `boss_adds`, and `trash` encounter generators |
| `src/stim/valuation.py` | Expected-value math (expected strike damage, DoT gain vs. clipping loss, buff/debuff value). Used by the greedy policy and planned as model features. |
| `src/stim/policies.py` | `RandomPolicy`, `GreedyPolicy` (class-agnostic), `AplPolicy` (interprets `[[apl]]`), `apl_namespace` |
| `tests/test_engine.py` | Every spec × scenario × policy runs; APL beats random; mechanics tests (energy ticks, CPs on target, finisher scaling, positionals, gap closer, rage, Heroic Strike, clone independence, determinism, schema errors) |
| `scripts/baselines.py` | Baseline DPS table |

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
- **Time to die.** `Enemy.time_to_die` is HP divided by an observed health-loss EMA (τ = 3 s),
  which is what a player could estimate.
- **Damage accounting** excludes overkill.
- **Speed.** About 35k-200k decisions per second in pure Python, depending on spec and policy.

### Baselines (30 fights each, mean DPS ± standard error)

| Spec | Scenario | random | greedy | APL |
| --- | --- | --- | --- | --- |
| Feral | boss | 478 ± 5 | **733 ± 10** | 684 ± 9 |
| Feral | boss_adds | 468 ± 6 | 648 ± 7 | 654 ± 10 |
| Feral | trash | 527 ± 8 | **844 ± 15** | 757 ± 13 |
| Rogue | boss | 387 ± 7 | 406 ± 9 | **447 ± 9** |
| Rogue | boss_adds | 388 ± 3 | 404 ± 5 | **444 ± 6** |
| Rogue | trash | 721 ± 23 | 900 ± 29 | 908 ± 28 |
| Warrior | boss | 258 ± 5 | 304 ± 7 | **386 ± 8** |
| Warrior | boss_adds | 294 ± 6 | 348 ± 7 | **423 ± 9** |
| Warrior | trash | 384 ± 17 | 525 ± 15 | 576 ± 19 |

On Feral, the class-agnostic greedy heuristic beats the hand-written priority list. The Feral APL
thresholds are probably mis-tuned; for example, it uses Shifting Power below 40 energy, while
guides say below 60.

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

## Next steps, in order

### 1. Teacher (`src/stim/teacher.py`)

- For a decision state with legal options A, run K samples (K ≈ 8). Each sample k gets its own
  future-event draw and RNG seed, and that draw is shared by every option (common random numbers).
- For each option a:
  1. clone the sim
  2. `resample_future` with sample k's draw
  3. `step(a)`
  4. follow the rollout policy until t0 + H (H ≈ 20 s)
- Score each rollout: Q_k(a) = damage gained + a terminal value. The terminal value is
  `resource_value` times unspent resources, plus pending DoT ticks that land before projected
  death.
- **Soft targets:** p(a) ∝ exp((Q̄(a) − max Q̄) / τ). Take τ from the standard error of paired
  differences, with a floor. Never use hard labels.
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
- **Cost estimate:** roughly 60-200 ms per state at K = 8 and H = 20 s. With 16 worker processes
  (the M5 Max has 18 cores), 40k states take about 5-10 minutes.

### 2. Features (`src/stim/features.py`)

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

### 3. Model (`src/stim/model.py`): Jev-style, about 0.5M parameters, target under 1 ms on CPU

- **Encoder:** d = 96, 4 heads, a 3-layer bidirectional transformer over all tokens with type
  embeddings. Option tokens also add the input embeddings of their ability and target tokens.
- **Readout:** 4 learned query slots per question and 2 cross-attention readout blocks.
- **Heads:**
  - Choice: `q · option / √d / learned temperature`, masked to legal options
  - Noul: will the target survive?
  - Score: ordinal stakes
  - Confidence: a sigmoid, trained to predict the probability mass on the teacher's best option

### 4. Loss: RLCD-style

- Choice: soft-target NLL + 0.5 × Brier + 0.2 × evidential term. The evidential term compares
  confidence to agreement with the teacher, with stop-gradient.
- Noul: BCE + Brier.
- Score: ordinal NLL.
- ECE is tracked as a metric only.

### 5. Data generation and training

Use multiprocess workers writing `.npz` shards under `runs/` (gitignored). Train with AdamW and a
cosine schedule. Split validation by episode. Do 2-3 rounds of expert iteration.

### 6. Evaluation

- Full-fight DPS on held-out seeds for random, greedy, APL, teacher, and student.
- Regret against the teacher on held-out labeled states. This is more meaningful than top-1
  agreement, since near-ties don't matter.
- Calibration: ECE and reliability.
- Latency: p50 and p99.
- Robustness: does the model adapt to perturbed tuning without retraining?

### 7. Practice trainer (the first UI)

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

### 8. Post-pull coach

Parse `WoWCombatLog.txt` (advanced logging), rebuild state at each GCD, and grade your choices
against the model. This needs a real Forever log to build against.

### 9. More specs, then casters and healers

- **Ret Paladin** needs seals as buffs with on-swing procs, and Judgement consuming a seal.
- **Enhancement Shaman** needs shared shock cooldowns (`cooldown_group` exists), Stormstrike
  charges (supported), and cast times (basic support exists).
- **Casters and healers** come after that.

## Known simplifications

- **Placeholders everywhere.** Every number in every spec is a placeholder. Rogue and Warrior
  `reference_dps` (800 / 900) don't match the sim's output (about 450 / 400), so recalibrate. It
  sets boss HP and the valuation scale.
- **No stances or forms.** Stance-gated abilities like Overpower aren't modeled.
- **Thin hit table.** No parry, block, or resource refunds on miss or dodge. DoTs don't crit.
  Refreshing a DoT resets it (vanilla-style, no pandemic).
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
  for the coach.
- What are your actual keybinds? The spec `key` fields are placeholders.
- Should talents be separate spec files (`feral_bleed.toml`, ...) or a `[talents]` section that
  patches the base spec?

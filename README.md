# stim

A simulation-trained rotation brain for World of Warcraft.

Hekili-style helpers are hand-written priority lists. stim works differently. A search-based
teacher simulates ahead to find the best ability in each situation, and a small Jev-style decision
model learns to make the same call in about a millisecond. The game rules are written down; the
strategy is learned.

First target: Feral Druid in WoW Forever. The engine is data-driven, so other classes are a spec file
away.

## Why it runs outside the game

WoW Forever ships Midnight's addon restrictions. In combat, health, auras, and even combo points come
back as secret values that addons can display but can't compute with. So stim is a practice trainer
(and later a post-pull coach built on combat logs). It never reads the game client or sends input to it.

## Status

Early scaffolding.

# Pressure Plate

Adapted from: https://github.com/uoe-agents/pressureplate

Checked against the upstream `main` branch on 2026-04-08.

Local notes:
- Rendering now uses `pygame` instead of `pyglet`.
- Environment mechanics live under [`impl/`](/home/omar/Projects/amethyst/csh/src/environments/pressure_plate/impl), with the package re-exporting the PettingZoo constructors from [`impl/env.py`](/home/omar/Projects/amethyst/csh/src/environments/pressure_plate/impl/env.py).
- The PettingZoo port keeps the original layouts, actions, door/plate mechanics, and reward shaping.
- The local port intentionally keeps dynamic agent/door observation layers in sync each step, while the upstream repo leaves those grid layers stale after `reset()`.

# Level-Based Foraging

Adapted from: https://github.com/semitable/lb-foraging

Local notes:
- Environment mechanics live under [`impl/`](/home/omar/Projects/amethyst/csh/src/environments/level_based_foraging/impl), with the package re-exporting PettingZoo constructors from [`impl/env.py`](/home/omar/Projects/amethyst/csh/src/environments/level_based_foraging/impl/env.py).
- The local version is a native PettingZoo `ParallelEnv`; it does not recreate the upstream Gymnasium registration matrix.
- Actions preserve upstream integer values: `NONE=0`, `NORTH=1`, `SOUTH=2`, `WEST=3`, `EAST=4`, `LOAD=5`.
- Rendering uses `pygame-ce` instead of upstream `pyglet`.
- Episode endings are split into PettingZoo semantics: collecting all food terminates the episode, while hitting the step limit truncates it.

## MIT License Notice

Copyright (c) 2021 Filippos Christianos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_ROOT) in sys.path:
    sys.path.remove(str(_VENDOR_ROOT))
sys.path.insert(0, str(_VENDOR_ROOT))

from flatland.envs.rail_env_action import RailEnvActions

from .impl.env import FlatlandParallelEnv, env, parallel_env, raw_env

__all__ = [
    "FlatlandParallelEnv",
    "RailEnvActions",
    "env",
    "parallel_env",
    "raw_env",
]

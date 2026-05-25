from __future__ import annotations

import functools
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec

from .warehouse import Warehouse


class RWAREParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for the RWARE environment.

    This wrapper keeps the original RWARE transition and observation logic intact,
    but exposes the environment through PettingZoo's simultaneous-action API.

    You can either:
      - wrap a registered Gymnasium id, e.g. ``env_name="rware-tiny-2ag-v2"``
      - or build directly from ``Warehouse(...)`` kwargs by passing ``env_name=None``
    """

    metadata = {
        "name": "rware_parallel_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        env_name: str | None = "rware-tiny-2ag-v2",
        render_mode: str | None = None,
        **env_kwargs: Any,
    ) -> None:
        self.env_name = env_name
        self.render_mode = render_mode
        self._env_kwargs = dict(env_kwargs)
        self.env = self._make_base_env(env_name=env_name, render_mode=render_mode, env_kwargs=env_kwargs)

        self.possible_agents = [
            f"agent_{i}" for i in range(self.env.unwrapped.n_agents)
        ]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []

        self._last_observations: dict[str, Any] = {}
        self._last_infos: dict[str, dict[str, Any]] = {}

        state_dim = sum(
            gym.spaces.flatdim(self.env.observation_space[i])
            for i in range(self.env.unwrapped.n_agents)
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

    @staticmethod
    def _make_base_env(
        env_name: str | None,
        render_mode: str | None,
        env_kwargs: dict[str, Any],
    ):
        if env_name is None:
            return Warehouse(render_mode=render_mode, **env_kwargs)
        return gym.make(env_name, render_mode=render_mode, **env_kwargs)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self.env.observation_space[self.agent_name_mapping[agent]]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self.env.action_space[self.agent_name_mapping[agent]]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]

        self._last_observations = {
            agent: observations[i] for i, agent in enumerate(self.agents)
        }
        shared_info = dict(info) if isinstance(info, dict) else {}
        self._last_infos = {agent: dict(shared_info) for agent in self.agents}
        return dict(self._last_observations), dict(self._last_infos)

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        missing = [agent for agent in self.agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")

        ordered_actions = [actions[agent] for agent in self.agents]
        observations, rewards, done, truncated, info = self.env.step(ordered_actions)

        obs_dict = {agent: observations[i] for i, agent in enumerate(self.agents)}
        rew_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}

        base_env = self.env.unwrapped
        max_steps_hit = (
            base_env.max_steps is not None
            and base_env._cur_steps >= base_env.max_steps
        )
        inactivity_hit = (
            base_env.max_inactivity_steps is not None
            and base_env._cur_inactive_steps >= base_env.max_inactivity_steps
        )

        # The original Gymnasium env collapses all episode endings into `done` and always
        # returns `truncated=False`. Here we restore more useful PettingZoo semantics:
        # - max_steps => truncation
        # - inactivity / task-defined ending => termination
        terminations = {
            agent: bool(done and (inactivity_hit or (not max_steps_hit and not truncated)))
            for agent in self.agents
        }
        truncations = {
            agent: bool(truncated or (done and max_steps_hit))
            for agent in self.agents
        }

        shared_info = dict(info) if isinstance(info, dict) else {}
        infos: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            agent_info = dict(shared_info)
            if done or truncated:
                agent_info["episode_done"] = True
                reasons = []
                if inactivity_hit:
                    reasons.append("inactivity")
                if max_steps_hit:
                    reasons.append("max_steps")
                if truncated and not max_steps_hit:
                    reasons.append("truncated")
                if reasons:
                    agent_info["episode_end_reason"] = "+".join(reasons)
            infos[agent] = agent_info

        self._last_observations = obs_dict
        self._last_infos = infos

        if done or truncated:
            self.agents = []

        return obs_dict, rew_dict, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self) -> np.ndarray:
        if not self._last_observations:
            raise RuntimeError("Call reset() before state().")

        flat_obs = []
        for agent in self.possible_agents:
            if agent in self._last_observations:
                flat_obs.append(
                    gym.spaces.flatten(
                        self.observation_space(agent), self._last_observations[agent]
                    )
                )

        if not flat_obs:
            return np.array([], dtype=np.float32)
        return np.concatenate(flat_obs).astype(np.float32, copy=False)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __str__(self) -> str:
        if self.env_name is None:
            return "rware_parallel"
        return f"rware_parallel<{self.env_name}>"



def parallel_env(**kwargs: Any) -> RWAREParallelEnv:
    return RWAREParallelEnv(**kwargs)



def env(**kwargs: Any):
    """AEC version produced from the simultaneous-action ParallelEnv."""
    return parallel_to_aec(parallel_env(**kwargs))

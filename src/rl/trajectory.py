from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class VectorEpisodicCostTracker:
    """Track completed per-agent episode costs across rollout boundaries.

    Rollout fragments are retained internally and never treated as complete
    episodes.  The most recent completed-episode estimate is retained for
    agents that do not finish an episode in the current rollout.
    """

    num_envs: int
    num_agents: int
    gamma: float = 1.0
    initial_returns: float | np.ndarray = 0.0
    _running_returns: np.ndarray = field(init=False, repr=False)
    _running_masses: np.ndarray = field(init=False, repr=False)
    _discounts: np.ndarray = field(init=False, repr=False)
    _last_returns: np.ndarray = field(init=False, repr=False)
    _last_scales: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.num_agents <= 0:
            raise ValueError("num_envs and num_agents must be positive.")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1].")

        shape = (int(self.num_envs), int(self.num_agents))
        self._running_returns = np.zeros(shape, dtype=np.float32)
        self._running_masses = np.zeros(shape, dtype=np.float32)
        self._discounts = np.ones(shape, dtype=np.float32)
        initial = np.asarray(self.initial_returns, dtype=np.float32)
        self._last_returns = np.broadcast_to(initial, (self.num_agents,)).copy()
        self._last_scales = np.ones((self.num_agents,), dtype=np.float32)

    def add(
        self,
        costs: np.ndarray,
        episode_dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add a rollout and return estimates, scales, and completion counts."""

        costs = np.asarray(costs, dtype=np.float32)
        episode_dones = np.asarray(episode_dones, dtype=np.float32)
        expected_tail = (self.num_envs, self.num_agents)
        if costs.shape != episode_dones.shape:
            raise ValueError("costs and episode_dones must have matching shapes.")
        if costs.ndim != 3 or costs.shape[1:] != expected_tail:
            raise ValueError("costs must have shape (steps, num_envs, num_agents).")

        completed_returns: list[list[float]] = [[] for _ in range(self.num_agents)]
        completed_masses: list[list[float]] = [[] for _ in range(self.num_agents)]

        for step_costs, step_dones in zip(costs, episode_dones, strict=True):
            self._running_returns += self._discounts * step_costs
            self._running_masses += self._discounts
            done_mask = step_dones.astype(bool)

            for env_idx, agent_idx in np.argwhere(done_mask):
                completed_returns[int(agent_idx)].append(
                    float(self._running_returns[env_idx, agent_idx])
                )
                completed_masses[int(agent_idx)].append(
                    float(self._running_masses[env_idx, agent_idx])
                )

            self._running_returns = np.where(done_mask, 0.0, self._running_returns)
            self._running_masses = np.where(done_mask, 0.0, self._running_masses)
            self._discounts = np.where(
                done_mask,
                1.0,
                self._discounts * float(self.gamma),
            ).astype(np.float32)

        completion_counts = np.asarray(
            [len(agent_returns) for agent_returns in completed_returns],
            dtype=np.int32,
        )
        for agent_idx, agent_returns in enumerate(completed_returns):
            if not agent_returns:
                continue
            self._last_returns[agent_idx] = float(np.mean(agent_returns))
            self._last_scales[agent_idx] = 1.0 / max(
                float(np.mean(completed_masses[agent_idx])),
                1e-8,
            )

        return (
            self._last_returns.copy(),
            self._last_scales.copy(),
            completion_counts,
        )


def episode_history_entry(
    *,
    episode: int,
    env_index: int,
    start_step: int,
    t_end: int,
    ep_len: int,
    returns: np.ndarray,
    agent_ids: tuple[str, ...],
    safety_violations: float,
) -> dict[str, object]:
    """Build one completed-episode log row shared by all trainers."""

    return_array = np.asarray(returns, dtype=float).reshape(-1)
    return {
        "episode": int(episode),
        "env_index": int(env_index),
        "start_step": int(start_step),
        "t_end": int(t_end),
        "ep_len": int(ep_len),
        "cum_reward": float(return_array.mean()) if return_array.size else 0.0,
        "cum_violations": float(safety_violations),
        "agent_returns": {
            str(agent_id): float(return_array[agent_idx])
            for agent_idx, agent_id in enumerate(agent_ids)
            if agent_idx < return_array.size
        },
    }


def compute_gae(
    *,
    reward: jax.Array,
    value: jax.Array,
    next_value: jax.Array,
    terminated: jax.Array,
    episode_done: jax.Array,
    gamma: jax.Array,
    gae_lambda: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """GAE with distinct termination and episode-boundary masks."""

    def _scan_step(
        gae: jax.Array,
        transition: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        step_reward, step_value, step_next_value, step_terminated, step_episode_done = (
            transition
        )
        bootstrap_mask = 1.0 - step_terminated
        recursion_mask = 1.0 - step_episode_done
        delta = step_reward + gamma * step_next_value * bootstrap_mask - step_value
        gae = delta + gamma * gae_lambda * recursion_mask * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _scan_step,
        jnp.zeros_like(next_value[-1]),
        (reward, value, next_value, terminated, episode_done),
        reverse=True,
    )
    return advantages, advantages + value


def discounted_cost_returns_at_starts(
    costs: np.ndarray,
    episode_dones: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    returns, _ = discounted_cost_returns_and_scales_at_starts(
        costs,
        episode_dones,
        gamma=gamma,
    )
    return returns


def discounted_cost_returns_and_scales_at_starts(
    costs: np.ndarray,
    episode_dones: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(costs, dtype=np.float32)
    episode_dones = np.asarray(episode_dones, dtype=np.float32)
    if costs.shape != episode_dones.shape:
        raise ValueError("costs and episode_dones must have matching shapes.")

    _, num_envs, num_agents = costs.shape
    returns_by_agent: list[list[float]] = [[] for _ in range(num_agents)]
    masses_by_agent: list[list[float]] = [[] for _ in range(num_agents)]

    for env_idx in range(num_envs):
        running_returns = np.zeros(num_agents, dtype=np.float32)
        discounted_masses = np.zeros(num_agents, dtype=np.float32)
        discounts = np.ones(num_agents, dtype=np.float32)
        segment_lengths = np.zeros(num_agents, dtype=np.int32)
        for step_costs, step_dones in zip(
            costs[:, env_idx],
            episode_dones[:, env_idx],
            strict=True,
        ):
            running_returns += discounts * step_costs
            discounted_masses += discounts
            segment_lengths += 1
            done_mask = step_dones.astype(bool)
            for agent_idx, done in enumerate(done_mask):
                if done:
                    returns_by_agent[agent_idx].append(
                        float(running_returns[agent_idx])
                    )
                    masses_by_agent[agent_idx].append(
                        float(discounted_masses[agent_idx])
                    )
                    running_returns[agent_idx] = 0.0
                    discounted_masses[agent_idx] = 0.0
                    discounts[agent_idx] = 1.0
                    segment_lengths[agent_idx] = 0
            discounts = np.where(done_mask, 1.0, discounts * float(gamma))

        for agent_idx, segment_length in enumerate(segment_lengths):
            if segment_length > 0:
                returns_by_agent[agent_idx].append(float(running_returns[agent_idx]))
                masses_by_agent[agent_idx].append(float(discounted_masses[agent_idx]))

    returns = np.asarray(
        [
            float(np.mean(agent_returns)) if agent_returns else 0.0
            for agent_returns in returns_by_agent
        ],
        dtype=np.float32,
    )
    scales = np.asarray(
        [
            1.0 / max(float(np.mean(agent_masses)), 1e-8) if agent_masses else 1.0
            for agent_masses in masses_by_agent
        ],
        dtype=np.float32,
    )
    return returns, scales


def clipped_value_loss(
    *,
    value: jax.Array,
    old_value: jax.Array,
    target: jax.Array,
    clip_eps: jax.Array,
    clip_value_loss: bool,
) -> jax.Array:
    value_losses = jnp.square(value - target)
    if not clip_value_loss:
        return 0.5 * value_losses.mean()

    value_pred_clipped = old_value + jnp.clip(
        value - old_value,
        -clip_eps,
        clip_eps,
    )
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


def normalize_advantage(advantage: jax.Array) -> jax.Array:
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8)


def ppo_approx_kl(log_ratio: jax.Array) -> jax.Array:
    ratio = jnp.exp(log_ratio)
    return jnp.mean((ratio - 1.0) - log_ratio)

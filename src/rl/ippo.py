from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any, Callable, Mapping, NamedTuple

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from pettingzoo import ParallelEnv

from .trajectory import (
    clipped_value_loss,
    compute_gae,
    episode_history_entry,
    normalize_advantage,
    ppo_approx_kl,
)


EnvFactory = Callable[..., ParallelEnv]
ProgressCallback = Callable[[Mapping[str, float]], None]


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_hidden = activation(actor_hidden)
        actor_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_hidden)
        actor_hidden = activation(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)

        critic_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_hidden = activation(critic_hidden)
        critic_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_hidden)
        critic_hidden = activation(critic_hidden)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden)

        return actor_logits, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    terminated: jax.Array
    episode_done: jax.Array
    action: jax.Array
    action_mask: jax.Array
    value: jax.Array
    next_value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array


class MultiAgentTrainState(NamedTuple):
    params: Any
    opt_state: Any


class EnvSpec(NamedTuple):
    agent_ids: tuple[str, ...]
    obs_dim: int
    action_dim: int
    observation_spaces: tuple[gym.Space, ...]


def resolve_env_factory(factory: EnvFactory | str) -> EnvFactory:
    if callable(factory):
        return factory

    if ":" not in factory:
        raise ValueError(
            "Environment factory must be callable or use 'module:function' syntax."
        )

    module_name, attr_name = factory.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr_name)
    if not callable(resolved):
        raise TypeError(f"Resolved environment factory '{factory}' is not callable.")
    return resolved


def validate_parallel_env(env: ParallelEnv) -> EnvSpec:
    agent_ids = tuple(env.possible_agents)
    if not agent_ids:
        raise ValueError("ParallelEnv must declare at least one possible agent.")

    observation_spaces = tuple(env.observation_space(agent) for agent in agent_ids)
    action_spaces = tuple(env.action_space(agent) for agent in agent_ids)

    first_obs_space = observation_spaces[0]
    first_action_space = action_spaces[0]

    if not all(isinstance(space, gym.spaces.Box) for space in observation_spaces):
        raise TypeError("IPPO v1 requires Box observation spaces for every agent.")
    if not all(isinstance(space, gym.spaces.Discrete) for space in action_spaces):
        raise TypeError("IPPO v1 requires Discrete action spaces for every agent.")

    obs_dim = gym.spaces.flatdim(first_obs_space)
    action_dim = first_action_space.n

    for agent, space in zip(agent_ids, observation_spaces, strict=True):
        if gym.spaces.flatdim(space) != obs_dim:
            raise ValueError(
                f"Agent '{agent}' observation space does not match the shared shape."
            )

    for agent, space in zip(agent_ids, action_spaces, strict=True):
        if space.n != action_dim:
            raise ValueError(
                f"Agent '{agent}' action space does not match the shared size."
            )

    return EnvSpec(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        observation_spaces=observation_spaces,
    )


def flatten_observation(space: gym.Space, observation: Any) -> np.ndarray:
    flat = gym.spaces.flatten(space, observation)
    return np.asarray(flat, dtype=np.float32).reshape(-1)


def dict_observations_to_array(
    observations: Mapping[str, Any],
    agent_ids: tuple[str, ...],
    observation_spaces: tuple[gym.Space, ...],
) -> np.ndarray:
    return np.stack(
        [
            flatten_observation(space, observations[agent])
            for agent, space in zip(agent_ids, observation_spaces, strict=True)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def dict_values_to_array(
    values: Mapping[str, Any],
    agent_ids: tuple[str, ...],
    dtype: np.dtype | type[np.floating] | type[np.integer] = np.float32,
) -> np.ndarray:
    return np.asarray([values[agent] for agent in agent_ids], dtype=dtype)


def dict_dones_to_array(
    terminations: Mapping[str, Any],
    truncations: Mapping[str, Any],
    agent_ids: tuple[str, ...],
) -> np.ndarray:
    terminated = dict_values_to_array(terminations, agent_ids, dtype=np.float32)
    truncated = dict_values_to_array(truncations, agent_ids, dtype=np.float32)
    return np.maximum(terminated, truncated)


def dict_infos_to_array(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    key: str,
    *,
    default: float = 0.0,
) -> np.ndarray:
    values: list[float] = []
    for agent in agent_ids:
        agent_info = infos.get(agent, {})
        raw_value = agent_info.get(key, default)
        if isinstance(raw_value, (bool, np.bool_)):
            values.append(float(raw_value))
        else:
            values.append(float(raw_value))
    return np.asarray(values, dtype=np.float32)


def action_masks_from_infos(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    action_dim: int,
) -> np.ndarray:
    masks: list[np.ndarray] = []
    for agent in agent_ids:
        raw_mask = infos.get(agent, {}).get("action_mask")
        if raw_mask is None:
            mask = np.ones(int(action_dim), dtype=bool)
        else:
            mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
            if mask.shape != (int(action_dim),):
                raise ValueError(
                    f"action_mask for {agent!r} has shape {mask.shape}, "
                    f"expected {(int(action_dim),)}."
                )
        if not bool(mask.any()):
            raise ValueError(f"action_mask for {agent!r} does not allow any action.")
        masks.append(mask)
    return np.stack(masks, axis=0).astype(bool, copy=False)


def assert_actions_respect_masks(
    actions: np.ndarray | jax.Array,
    action_masks: np.ndarray | jax.Array,
    agent_ids: tuple[str, ...],
    *,
    phase: str,
    env_index: int | None = None,
    global_step: int | None = None,
    update_idx: int | None = None,
) -> None:
    action_array = np.asarray(actions, dtype=np.int64).reshape(-1)
    mask_array = np.asarray(action_masks, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError(
            f"Action mask assertion during {phase} expected a 2-D mask array, "
            f"got shape {mask_array.shape}."
        )
    expected_agents = len(agent_ids)
    if action_array.shape != (expected_agents,):
        raise ValueError(
            f"Action mask assertion during {phase} expected actions shape "
            f"{(expected_agents,)}, got {action_array.shape}."
        )
    if mask_array.shape[0] != expected_agents:
        raise ValueError(
            f"Action mask assertion during {phase} expected {expected_agents} "
            f"agent masks, got shape {mask_array.shape}."
        )
    if not bool(mask_array.any(axis=1).all()):
        empty_agents = [
            agent_id
            for agent_idx, agent_id in enumerate(agent_ids)
            if not bool(mask_array[agent_idx].any())
        ]
        raise ValueError(
            f"Action mask assertion during {phase} found empty masks for "
            f"agents {empty_agents!r}; env_index={env_index!r}; "
            f"global_step={global_step!r}; update_idx={update_idx!r}."
        )

    for agent_idx, agent_id in enumerate(agent_ids):
        action = int(action_array[agent_idx])
        allowed = np.flatnonzero(mask_array[agent_idx]).astype(int).tolist()
        if (
            action < 0
            or action >= mask_array.shape[1]
            or not bool(mask_array[agent_idx, action])
        ):
            raise ValueError(
                f"Action mask assertion failed during {phase}: "
                f"env_index={env_index!r}; agent_id={agent_id!r}; "
                f"action={action}; allowed actions={allowed}; "
                f"global_step={global_step!r}; update_idx={update_idx!r}."
            )


def all_true_action_masks(
    num_envs: int,
    num_agents: int,
    action_dim: int,
) -> np.ndarray:
    return np.ones((num_envs, num_agents, action_dim), dtype=bool)


def mask_logits(logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    return jnp.where(action_mask.astype(bool), logits, jnp.full_like(logits, -1.0e9))


def masked_argmax(values: jax.Array, action_mask: jax.Array) -> jax.Array:
    masked_values = jnp.where(
        action_mask.astype(bool),
        values,
        jnp.full_like(values, -1.0e9),
    )
    return jnp.argmax(masked_values, axis=-1)


def masked_max(values: jax.Array, action_mask: jax.Array) -> jax.Array:
    return jnp.max(
        jnp.where(
            action_mask.astype(bool),
            values,
            jnp.full_like(values, -1.0e9),
        ),
        axis=-1,
    )


def sample_masked_random_actions(action_mask: jax.Array, rng: jax.Array) -> jax.Array:
    logits = jnp.where(
        action_mask.astype(bool),
        jnp.zeros(action_mask.shape, dtype=jnp.float32),
        jnp.full(action_mask.shape, -1.0e9, dtype=jnp.float32),
    )
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)


def shared_info_value(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    key: str,
    *,
    default: float = 0.0,
) -> float:
    for agent in agent_ids:
        agent_info = infos.get(agent, {})
        if key in agent_info:
            raw_value = agent_info[key]
            if isinstance(raw_value, (bool, np.bool_)):
                return float(raw_value)
            return float(raw_value)
    return float(default)


def actions_array_to_dict(
    actions: np.ndarray | jax.Array,
    agent_ids: tuple[str, ...],
) -> dict[str, int]:
    action_array = np.asarray(actions, dtype=np.int32)
    return {
        agent: int(action_array[idx])
        for idx, agent in enumerate(agent_ids)
    }


def init_train_state(
    rng: jax.Array,
    network: ActorCritic,
    tx: optax.GradientTransformation,
    num_agents: int,
    obs_dim: int,
) -> MultiAgentTrainState:
    init_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    init_rngs = jax.random.split(rng, num_agents)
    params = jax.vmap(network.init)(init_rngs, init_obs)
    opt_state = jax.vmap(tx.init)(params)
    return MultiAgentTrainState(params=params, opt_state=opt_state)


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    config = dict(config)
    config.setdefault("CLIP_VALUE_LOSS", True)
    config.setdefault("VALUE_CLIP_EPS", config["CLIP_EPS"])
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = (
        int(config["NUM_STEPS"]) * int(config["NUM_ENVS"]) // int(config["NUM_MINIBATCHES"])
    )

    factory_ref = env_factory or config.get("ENV_FACTORY") or config.get("ENV_NAME")
    if factory_ref is None:
        raise ValueError(
            "Provide an env_factory argument or set ENV_FACTORY/ENV_NAME in config."
        )
    resolved_env_factory = resolve_env_factory(factory_ref)
    env_kwargs = dict(config.get("ENV_KWARGS", {}))

    probe_env = resolved_env_factory(**env_kwargs)
    try:
        env_spec = validate_parallel_env(probe_env)
    finally:
        probe_env.close()

    network = ActorCritic(
        env_spec.action_dim,
        activation=config.get("ACTIVATION", "tanh"),
    )

    def linear_schedule(step_count: jax.Array) -> jax.Array:
        minibatches_per_update = int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])
        frac = 1.0 - (step_count // minibatches_per_update) / max(config["NUM_UPDATES"], 1)
        return jnp.asarray(config["LR"], dtype=jnp.float32) * frac

    if config.get("ANNEAL_LR", False):
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    def apply_model(params: Any, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(params, obs)

    sample_action_jit = jax.jit(
        lambda params, obs, action_mask, rng: _sample_actions(
            apply_model,
            params,
            obs,
            action_mask,
            rng,
        )
    )
    last_value_jit = jax.jit(
        lambda params, obs: _compute_last_values(apply_model, params, obs)
    )
    compute_gae_jit = jax.jit(
        lambda traj_batch: _compute_gae(config, traj_batch)
    )
    update_minibatch_jit = jax.jit(
        lambda state, minibatch: _update_minibatch(config, network, tx, state, minibatch)
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        rng = jax.random.PRNGKey(int(rng)) if np.isscalar(rng) else rng
        num_envs = int(config["NUM_ENVS"])
        num_steps = int(config["NUM_STEPS"])
        num_agents = len(env_spec.agent_ids)
        minibatch_size = int(config["MINIBATCH_SIZE"])
        batch_size = num_steps * num_envs

        if batch_size % int(config["NUM_MINIBATCHES"]) != 0:
            raise ValueError(
                "NUM_STEPS * NUM_ENVS must be divisible by NUM_MINIBATCHES."
            )

        envs = [resolved_env_factory(**env_kwargs) for _ in range(num_envs)]
        episode_returns = np.zeros((num_envs, num_agents), dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        episode_start_steps = np.zeros(num_envs, dtype=np.int64)
        reset_counter = 0
        total_completed_episodes = 0
        metrics_history: dict[str, list[float]] = defaultdict(list)
        episode_history: list[dict[str, object]] = []

        try:
            rng, init_rng = jax.random.split(rng)
            train_state = init_train_state(
                init_rng,
                network,
                tx,
                num_agents=num_agents,
                obs_dim=env_spec.obs_dim,
            )

            last_obs = np.zeros(
                (num_envs, num_agents, env_spec.obs_dim),
                dtype=np.float32,
            )
            last_action_masks = all_true_action_masks(
                num_envs,
                num_agents,
                env_spec.action_dim,
            )
            for env_idx, env in enumerate(envs):
                obs_dict, infos = env.reset(seed=int(config.get("SEED", 0)) + reset_counter)
                reset_counter += 1
                last_obs[env_idx] = dict_observations_to_array(
                    obs_dict,
                    env_spec.agent_ids,
                    env_spec.observation_spaces,
                )
                last_action_masks[env_idx] = action_masks_from_infos(
                    infos,
                    env_spec.agent_ids,
                    env_spec.action_dim,
                )

            for update_idx in range(int(config["NUM_UPDATES"])):
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_actions = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.int32,
                )
                traj_action_masks = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.action_dim),
                    dtype=bool,
                )
                traj_log_probs = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_next_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_rewards = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_safety_violation_counts = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_safety_violation_fractions = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_shield_intervention_counts = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_shield_intervention_fractions = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                latest_cumulative_safety_violations = np.zeros(
                    num_envs,
                    dtype=np.float32,
                )
                traj_terminated = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_episode_dones = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                completed_returns: list[np.ndarray] = []
                completed_lengths: list[int] = []
                completed_safety_violation_counts: list[float] = []

                for step_idx in range(num_steps):
                    obs_batch = jnp.asarray(last_obs)
                    action_mask_batch = jnp.asarray(last_action_masks)
                    rng, sample_rng = jax.random.split(rng)
                    actions, log_probs, values = sample_action_jit(
                        train_state.params,
                        obs_batch,
                        action_mask_batch,
                        sample_rng,
                    )
                    actions_np = np.asarray(actions)
                    log_probs_np = np.asarray(log_probs)
                    values_np = np.asarray(values)

                    next_obs = np.zeros_like(last_obs)
                    next_obs_for_value = np.zeros_like(last_obs)
                    reward_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    terminated_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    episode_done_batch = np.zeros(
                        (num_envs, num_agents),
                        dtype=np.float32,
                    )

                    for env_idx, env in enumerate(envs):
                        env_global_step = (
                            update_idx * batch_size
                            + step_idx * num_envs
                            + env_idx
                            + 1
                        )
                        assert_actions_respect_masks(
                            actions_np[env_idx],
                            last_action_masks[env_idx],
                            env_spec.agent_ids,
                            phase="IPPO rollout sampling",
                            env_index=env_idx,
                            global_step=env_global_step,
                            update_idx=update_idx,
                        )
                        action_dict = actions_array_to_dict(
                            actions_np[env_idx],
                            env_spec.agent_ids,
                        )
                        (
                            obs_dict,
                            rewards,
                            terminations,
                            truncations,
                            infos,
                        ) = env.step(action_dict)

                        reward_array = dict_values_to_array(
                            rewards,
                            env_spec.agent_ids,
                            dtype=np.float32,
                        )
                        terminated_array = dict_values_to_array(
                            terminations,
                            env_spec.agent_ids,
                            dtype=np.float32,
                        )
                        truncated_array = dict_values_to_array(
                            truncations,
                            env_spec.agent_ids,
                            dtype=np.float32,
                        )
                        done_array = np.maximum(terminated_array, truncated_array)
                        safety_violation_count = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "safety_violations_mean",
                            default=0.0,
                        )
                        safety_violation_fraction = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "safety_violations_agent_fraction_mean",
                            default=0.0,
                        )
                        shield_intervention_count = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "shield_interventions_mean",
                            default=0.0,
                        )
                        shield_intervention_fraction = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "shield_interventions_agent_fraction_mean",
                            default=0.0,
                        )
                        if not np.all(done_array == done_array[0]):
                            raise ValueError(
                                "IPPO v1 expects all agents in an environment to finish together."
                            )

                        reward_batch[env_idx] = reward_array
                        terminated_batch[env_idx] = terminated_array
                        episode_done_batch[env_idx] = done_array
                        traj_safety_violation_counts[step_idx, env_idx] = (
                            safety_violation_count
                        )
                        traj_safety_violation_fractions[step_idx, env_idx] = (
                            safety_violation_fraction
                        )
                        traj_shield_intervention_counts[step_idx, env_idx] = (
                            shield_intervention_count
                        )
                        traj_shield_intervention_fractions[step_idx, env_idx] = (
                            shield_intervention_fraction
                        )
                        latest_cumulative_safety_violations[env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "safety_violations_cumulative",
                            default=0.0,
                        )
                        episode_returns[env_idx] += reward_array
                        episode_lengths[env_idx] += 1
                        if all(agent in obs_dict for agent in env_spec.agent_ids):
                            next_obs_for_value[env_idx] = dict_observations_to_array(
                                obs_dict,
                                env_spec.agent_ids,
                                env_spec.observation_spaces,
                            )

                        if bool(done_array[0]):
                            completed_returns.append(episode_returns[env_idx].copy())
                            completed_lengths.append(int(episode_lengths[env_idx]))
                            episode_safety_violations = shared_info_value(
                                infos,
                                env_spec.agent_ids,
                                "episode_safety_violations_mean",
                                default=0.0,
                            )
                            completed_safety_violation_counts.append(
                                episode_safety_violations
                            )
                            episode_history.append(
                                episode_history_entry(
                                    episode=total_completed_episodes + 1,
                                    env_index=env_idx,
                                    start_step=int(episode_start_steps[env_idx]),
                                    t_end=env_global_step,
                                    ep_len=int(episode_lengths[env_idx]),
                                    returns=episode_returns[env_idx],
                                    agent_ids=env_spec.agent_ids,
                                    safety_violations=episode_safety_violations,
                                )
                            )
                            total_completed_episodes += 1
                            obs_dict, infos = env.reset(
                                seed=int(config.get("SEED", 0)) + reset_counter
                            )
                            reset_counter += 1
                            episode_returns[env_idx].fill(0.0)
                            episode_lengths[env_idx] = 0
                            episode_start_steps[env_idx] = env_global_step

                        next_obs[env_idx] = dict_observations_to_array(
                            obs_dict,
                            env_spec.agent_ids,
                            env_spec.observation_spaces,
                        )
                        last_action_masks[env_idx] = action_masks_from_infos(
                            infos,
                            env_spec.agent_ids,
                            env_spec.action_dim,
                        )

                    traj_obs[step_idx] = last_obs
                    traj_actions[step_idx] = actions_np
                    traj_action_masks[step_idx] = np.asarray(action_mask_batch)
                    traj_log_probs[step_idx] = log_probs_np
                    traj_values[step_idx] = values_np
                    traj_next_values[step_idx] = np.asarray(
                        last_value_jit(train_state.params, jnp.asarray(next_obs_for_value))
                    )
                    traj_rewards[step_idx] = reward_batch
                    traj_terminated[step_idx] = terminated_batch
                    traj_episode_dones[step_idx] = episode_done_batch
                    last_obs = next_obs

                traj_batch = Transition(
                    terminated=jnp.asarray(traj_terminated),
                    episode_done=jnp.asarray(traj_episode_dones),
                    action=jnp.asarray(traj_actions),
                    action_mask=jnp.asarray(traj_action_masks),
                    value=jnp.asarray(traj_values),
                    next_value=jnp.asarray(traj_next_values),
                    reward=jnp.asarray(traj_rewards),
                    log_prob=jnp.asarray(traj_log_probs),
                    obs=jnp.asarray(traj_obs),
                )

                advantages, targets = compute_gae_jit(traj_batch)

                flat_batch = {
                    "obs": jnp.transpose(traj_batch.obs, (2, 0, 1, 3)).reshape(
                        num_agents, batch_size, env_spec.obs_dim
                    ),
                    "action": jnp.transpose(traj_batch.action, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "action_mask": jnp.transpose(
                        traj_batch.action_mask,
                        (2, 0, 1, 3),
                    ).reshape(num_agents, batch_size, env_spec.action_dim),
                    "log_prob": jnp.transpose(traj_batch.log_prob, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "value": jnp.transpose(traj_batch.value, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "advantage": jnp.transpose(advantages, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "target": jnp.transpose(targets, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                }

                minibatch_metrics: list[dict[str, np.ndarray]] = []
                for _ in range(int(config["UPDATE_EPOCHS"])):
                    rng, perm_rng = jax.random.split(rng)
                    permutation = jax.random.permutation(perm_rng, batch_size)
                    shuffled = {
                        key: jnp.take(value, permutation, axis=1)
                        for key, value in flat_batch.items()
                    }
                    for batch_start in range(0, batch_size, minibatch_size):
                        minibatch = {
                            key: value[:, batch_start : batch_start + minibatch_size]
                            for key, value in shuffled.items()
                        }
                        train_state, loss_info = update_minibatch_jit(
                            train_state,
                            minibatch,
                        )
                        minibatch_metrics.append(
                            {
                                name: np.asarray(metric)
                                for name, metric in loss_info.items()
                            }
                        )

                aggregated_loss_metrics: dict[str, np.ndarray] = {}
                for key in minibatch_metrics[0]:
                    aggregated_loss_metrics[key] = np.stack(
                        [metric[key] for metric in minibatch_metrics],
                        axis=0,
                    ).mean(axis=0)

                step_reward_mean = traj_rewards.mean(axis=(0, 1))
                safety_violations_mean = float(traj_safety_violation_counts.mean())
                safety_violations_agent_fraction_mean = float(
                    traj_safety_violation_fractions.mean()
                )
                shield_interventions_mean = float(traj_shield_intervention_counts.mean())
                shield_interventions_agent_fraction_mean = float(
                    traj_shield_intervention_fractions.mean()
                )
                cumulative_safety_violations = float(
                    latest_cumulative_safety_violations.sum()
                )
                if completed_returns:
                    mean_episode_return = np.stack(completed_returns, axis=0).mean(axis=0)
                    mean_episode_length = float(np.mean(completed_lengths))
                    mean_episode_safety_violations = float(
                        np.mean(completed_safety_violation_counts)
                    )
                else:
                    mean_episode_return = np.zeros(num_agents, dtype=np.float32)
                    mean_episode_length = 0.0
                    mean_episode_safety_violations = 0.0

                metrics = {
                    "update": float(update_idx),
                    "global_step": float((update_idx + 1) * batch_size),
                    "completed_episodes": float(total_completed_episodes),
                    "episode_length": mean_episode_length,
                    "reward_mean": float(step_reward_mean.mean()),
                    "episode_return_mean": float(mean_episode_return.mean()),
                    "safety_violations_mean": safety_violations_mean,
                    "safety_violations_agent_fraction_mean": (
                        safety_violations_agent_fraction_mean
                    ),
                    "safety_violations_cumulative": cumulative_safety_violations,
                    "episode_safety_violations_mean": mean_episode_safety_violations,
                    "shield_interventions_mean": shield_interventions_mean,
                    "shield_interventions_agent_fraction_mean": (
                        shield_interventions_agent_fraction_mean
                    ),
                }

                for key, value in aggregated_loss_metrics.items():
                    metrics[key] = float(value.mean())
                for agent_idx, agent_id in enumerate(env_spec.agent_ids):
                    metrics[f"{agent_id}/reward_mean"] = float(step_reward_mean[agent_idx])
                    metrics[f"{agent_id}/episode_return"] = float(
                        mean_episode_return[agent_idx]
                    )
                    for key, value in aggregated_loss_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f}"
                    )

                if progress_callback is not None:
                    progress_callback(
                        {
                            "update_idx": float(update_idx),
                            "num_updates": float(config["NUM_UPDATES"]),
                            "global_step": float(metrics["global_step"]),
                            "total_timesteps": float(config["TOTAL_TIMESTEPS"]),
                            "progress_fraction": float(
                                (update_idx + 1) / max(int(config["NUM_UPDATES"]), 1)
                            ),
                        }
                    )

                for key, value in metrics.items():
                    metrics_history[key].append(float(value))

            return {
                "train_state": train_state,
                "agent_ids": env_spec.agent_ids,
                "metrics": {
                    key: jnp.asarray(values)
                    for key, values in metrics_history.items()
                },
                "episode_history": episode_history,
            }
        finally:
            for env in envs:
                env.close()

    return train


def _sample_actions(
    apply_model: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array]],
    params: Any,
    obs: jax.Array,
    action_mask: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    mask_by_agent = jnp.swapaxes(action_mask, 0, 1)
    logits, values = apply_model(params, obs_by_agent)
    logits = mask_logits(logits, mask_by_agent)
    sample_rngs = jax.random.split(rng, logits.shape[0])

    def _sample(agent_logits: jax.Array, agent_rng: jax.Array):
        pi = distrax.Categorical(logits=agent_logits)
        actions = pi.sample(seed=agent_rng)
        log_probs = pi.log_prob(actions)
        return actions, log_probs

    actions, log_probs = jax.vmap(_sample)(logits, sample_rngs)
    return (
        jnp.swapaxes(actions, 0, 1),
        jnp.swapaxes(log_probs, 0, 1),
        jnp.swapaxes(values, 0, 1),
    )


def _compute_last_values(
    apply_model: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array]],
    params: Any,
    obs: jax.Array,
) -> jax.Array:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    _, values = apply_model(params, obs_by_agent)
    return jnp.swapaxes(values, 0, 1)


def _compute_gae(
    config: Mapping[str, Any],
    traj_batch: Transition,
) -> tuple[jax.Array, jax.Array]:
    return compute_gae(
        reward=traj_batch.reward,
        value=traj_batch.value,
        next_value=traj_batch.next_value,
        terminated=traj_batch.terminated,
        episode_done=traj_batch.episode_done,
        gamma=jnp.asarray(config["GAMMA"], dtype=jnp.float32),
        gae_lambda=jnp.asarray(config["GAE_LAMBDA"], dtype=jnp.float32),
    )


def _update_minibatch(
    config: Mapping[str, Any],
    network: ActorCritic,
    tx: optax.GradientTransformation,
    train_state: MultiAgentTrainState,
    minibatch: Mapping[str, jax.Array],
) -> tuple[MultiAgentTrainState, dict[str, jax.Array]]:
    clip_eps = jnp.asarray(config["CLIP_EPS"], dtype=jnp.float32)
    value_clip_eps = jnp.asarray(config["VALUE_CLIP_EPS"], dtype=jnp.float32)
    ent_coef = jnp.asarray(config["ENT_COEF"], dtype=jnp.float32)
    vf_coef = jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", True))

    def _loss_fn(
        params: Any,
        obs: jax.Array,
        action: jax.Array,
        action_mask: jax.Array,
        old_log_prob: jax.Array,
        old_value: jax.Array,
        advantage: jax.Array,
        target: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits, value = network.apply(params, obs)
        logits = mask_logits(logits, action_mask)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(action)

        value_loss = clipped_value_loss(
            value=value,
            old_value=old_value,
            target=target,
            clip_eps=value_clip_eps,
            clip_value_loss=clip_value_loss,
        )

        normalized_advantage = normalize_advantage(advantage)
        log_ratio = log_prob - old_log_prob
        ratio = jnp.exp(log_ratio)
        unclipped = ratio * normalized_advantage
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_advantage
        actor_loss = -jnp.minimum(unclipped, clipped).mean()
        entropy = pi.entropy().mean()
        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

        approx_kl = ppo_approx_kl(log_ratio)
        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": value_loss,
            "entropy": entropy,
            "ratio": ratio.mean(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    grad_fn = jax.vmap(
        jax.value_and_grad(_loss_fn, has_aux=True),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
    (losses, loss_info), grads = grad_fn(
        train_state.params,
        minibatch["obs"],
        minibatch["action"],
        minibatch["action_mask"],
        minibatch["log_prob"],
        minibatch["value"],
        minibatch["advantage"],
        minibatch["target"],
    )

    updates, next_opt_state = jax.vmap(tx.update)(
        grads,
        train_state.opt_state,
        train_state.params,
    )
    next_params = jax.vmap(optax.apply_updates)(train_state.params, updates)

    metrics = {
        key: value
        for key, value in loss_info.items()
    }
    metrics["total_loss"] = losses

    return MultiAgentTrainState(params=next_params, opt_state=next_opt_state), metrics


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 64,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "SEED": 0,
        "DEBUG": True,
        "ENV_FACTORY": "environments.pressure_plate:parallel_env",
        "ENV_KWARGS": {
            "height": 15,
            "width": 9,
            "n_agents": 4,
            "sensor_range": 3,
            "layout": "linear",
            "max_cycles": 8,
        },
    }

    rng = jax.random.PRNGKey(config["SEED"])
    trainer = make_train(config)
    output = trainer(rng)
    print(output["metrics"]["episode_return_mean"])

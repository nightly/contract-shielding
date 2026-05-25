from __future__ import annotations

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

from .ippo import (
    EnvFactory,
    ProgressCallback,
    assert_actions_respect_masks,
    actions_array_to_dict,
    action_masks_from_infos,
    all_true_action_masks,
    dict_observations_to_array,
    dict_values_to_array,
    mask_logits,
    resolve_env_factory,
    shared_info_value,
    validate_parallel_env,
)
from .trajectory import (
    clipped_value_loss,
    compute_gae,
    episode_history_entry,
    normalize_advantage,
    ppo_approx_kl,
)


class Actor(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        hidden = activation(hidden)
        hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(hidden)
        hidden = activation(hidden)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)


class CentralCritic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        hidden = activation(hidden)
        hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(hidden)
        hidden = activation(hidden)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(hidden)
        return jnp.squeeze(value, axis=-1)


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
    state: jax.Array


class MAPPOTrainState(NamedTuple):
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any


class MAPPOEnvSpec(NamedTuple):
    agent_ids: tuple[str, ...]
    obs_dim: int
    action_dim: int
    state_dim: int
    observation_spaces: tuple[gym.Space, ...]


def _state_space(
    env: ParallelEnv,
    *,
    algorithm_name: str = "MAPPO",
) -> gym.Space | None:
    raw_state_space = getattr(env, "state_space", None)
    if callable(raw_state_space):
        raw_state_space = raw_state_space()
    if raw_state_space is None:
        return None
    if not isinstance(raw_state_space, gym.Space):
        raise TypeError(
            f"{algorithm_name} requires env.state_space to be a gymnasium Space when provided."
        )
    return raw_state_space


def flatten_global_state(
    env: ParallelEnv,
    *,
    expected_dim: int | None = None,
    validate_state_space: bool = True,
    algorithm_name: str = "MAPPO",
) -> np.ndarray:
    state_fn = getattr(env, "state", None)
    if not callable(state_fn):
        raise TypeError(f"{algorithm_name} requires environments to expose env.state().")

    try:
        raw_state = state_fn()
    except NotImplementedError as exc:
        raise TypeError(f"{algorithm_name} requires env.state() to be implemented.") from exc
    if raw_state is None:
        raise TypeError(
            f"{algorithm_name} requires env.state() to return a numeric array, not None."
        )

    state_array = np.asarray(raw_state)
    if not (
        np.issubdtype(state_array.dtype, np.number)
        or np.issubdtype(state_array.dtype, np.bool_)
    ):
        raise TypeError(
            f"{algorithm_name} requires env.state() to return a numeric array, "
            f"got dtype {state_array.dtype}."
        )

    flat_state = np.asarray(state_array, dtype=np.float32).reshape(-1)
    if flat_state.size <= 0:
        raise ValueError(f"{algorithm_name} requires env.state() to return a non-empty array.")
    if expected_dim is not None and flat_state.size != int(expected_dim):
        raise ValueError(
            f"{algorithm_name} requires env.state() to keep a consistent flattened length; "
            f"expected {int(expected_dim)}, got {flat_state.size}."
        )
    if not np.isfinite(flat_state).all():
        raise ValueError(f"{algorithm_name} requires env.state() to contain only finite values.")
    if validate_state_space:
        state_space = _state_space(env, algorithm_name=algorithm_name)
        if state_space is not None:
            state_space_dim = gym.spaces.flatdim(state_space)
            if state_space_dim != flat_state.size:
                raise ValueError(
                    f"{algorithm_name} requires env.state_space to match env.state(); "
                    f"state_space flatdim is {state_space_dim}, state length is "
                    f"{flat_state.size}."
                )
            if isinstance(state_space, gym.spaces.Box):
                space_state = np.asarray(raw_state, dtype=state_space.dtype)
            else:
                space_state = raw_state
            if not state_space.contains(space_state):
                raise ValueError(
                    f"{algorithm_name} requires env.state() to be contained in state_space."
                )
    return flat_state


def validate_mappo_env(env: ParallelEnv) -> MAPPOEnvSpec:
    base_spec = validate_parallel_env(env)
    env.reset(seed=0)
    state = flatten_global_state(env)
    return MAPPOEnvSpec(
        agent_ids=base_spec.agent_ids,
        obs_dim=base_spec.obs_dim,
        action_dim=base_spec.action_dim,
        state_dim=int(state.shape[0]),
        observation_spaces=base_spec.observation_spaces,
    )


def init_train_state(
    rng: jax.Array,
    actor: Actor,
    critic: CentralCritic,
    tx: optax.GradientTransformation,
    *,
    num_agents: int,
    obs_dim: int,
    state_dim: int,
) -> MAPPOTrainState:
    actor_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    critic_state = jnp.zeros((num_agents, state_dim), dtype=jnp.float32)
    actor_rng, critic_rng = jax.random.split(rng)
    actor_params = jax.vmap(actor.init)(
        jax.random.split(actor_rng, num_agents),
        actor_obs,
    )
    critic_params = jax.vmap(critic.init)(
        jax.random.split(critic_rng, num_agents),
        critic_state,
    )
    actor_opt_state = jax.vmap(tx.init)(actor_params)
    critic_opt_state = jax.vmap(tx.init)(critic_params)
    return MAPPOTrainState(
        actor_params=actor_params,
        critic_params=critic_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
    )


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    config = dict(config)
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
        env_spec = validate_mappo_env(probe_env)
    finally:
        probe_env.close()

    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_EPS"] = float(config["CLIP_EPS"]) / float(len(env_spec.agent_ids))
    config.setdefault("CLIP_VALUE_LOSS", True)
    config.setdefault("VALUE_CLIP_EPS", config["CLIP_EPS"])

    actor = Actor(
        env_spec.action_dim,
        activation=config.get("ACTIVATION", "tanh"),
    )
    critic = CentralCritic(activation=config.get("ACTIVATION", "tanh"))

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

    def apply_actor(params: Any, obs: jax.Array) -> jax.Array:
        return jax.vmap(lambda p, o: actor.apply(p, o), in_axes=(0, 0))(params, obs)

    def apply_critic(params: Any, state: jax.Array) -> jax.Array:
        values_by_agent = jax.vmap(
            lambda p: critic.apply(p, state),
            in_axes=0,
        )(params)
        return jnp.swapaxes(values_by_agent, 0, 1)

    sample_action_jit = jax.jit(
        lambda actor_params, critic_params, obs, action_mask, state, rng: _sample_actions(
            apply_actor,
            apply_critic,
            actor_params,
            critic_params,
            obs,
            action_mask,
            state,
            rng,
        )
    )
    last_value_jit = jax.jit(
        lambda critic_params, state: _compute_last_values(
            apply_critic,
            critic_params,
            state,
        )
    )
    compute_gae_jit = jax.jit(lambda traj_batch: _compute_gae(config, traj_batch))
    update_minibatch_jit = jax.jit(
        lambda train_state, minibatch: _update_minibatch(
            config,
            actor,
            critic,
            tx,
            train_state,
            minibatch,
        )
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
                actor,
                critic,
                tx,
                num_agents=num_agents,
                obs_dim=env_spec.obs_dim,
                state_dim=env_spec.state_dim,
            )

            last_obs = np.zeros(
                (num_envs, num_agents, env_spec.obs_dim),
                dtype=np.float32,
            )
            last_state = np.zeros((num_envs, env_spec.state_dim), dtype=np.float32)
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
                last_state[env_idx] = flatten_global_state(
                    env,
                    expected_dim=env_spec.state_dim,
                    validate_state_space=False,
                )

            for update_idx in range(int(config["NUM_UPDATES"])):
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_states = np.zeros(
                    (num_steps, num_envs, env_spec.state_dim),
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
                    state_batch = jnp.asarray(last_state)
                    rng, sample_rng = jax.random.split(rng)
                    actions, log_probs, values = sample_action_jit(
                        train_state.actor_params,
                        train_state.critic_params,
                        obs_batch,
                        action_mask_batch,
                        state_batch,
                        sample_rng,
                    )
                    actions_np = np.asarray(actions)
                    log_probs_np = np.asarray(log_probs)
                    values_np = np.asarray(values)

                    next_obs = np.zeros_like(last_obs)
                    next_state = np.zeros_like(last_state)
                    next_state_for_value = np.zeros_like(last_state)
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
                            phase="MAPPO rollout sampling",
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
                                "MAPPO expects all agents in an environment to finish together."
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
                            next_obs[env_idx] = dict_observations_to_array(
                                obs_dict,
                                env_spec.agent_ids,
                                env_spec.observation_spaces,
                            )
                        next_state_for_value[env_idx] = flatten_global_state(
                            env,
                            expected_dim=env_spec.state_dim,
                            validate_state_space=False,
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
                        next_state[env_idx] = flatten_global_state(
                            env,
                            expected_dim=env_spec.state_dim,
                            validate_state_space=False,
                        )

                    traj_obs[step_idx] = last_obs
                    traj_states[step_idx] = last_state
                    traj_actions[step_idx] = actions_np
                    traj_action_masks[step_idx] = np.asarray(action_mask_batch)
                    traj_log_probs[step_idx] = log_probs_np
                    traj_values[step_idx] = values_np
                    traj_next_values[step_idx] = np.asarray(
                        last_value_jit(
                            train_state.critic_params,
                            jnp.asarray(next_state_for_value),
                        )
                    )
                    traj_rewards[step_idx] = reward_batch
                    traj_terminated[step_idx] = terminated_batch
                    traj_episode_dones[step_idx] = episode_done_batch
                    last_obs = next_obs
                    last_state = next_state

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
                    state=jnp.asarray(traj_states),
                )

                advantages, targets = compute_gae_jit(traj_batch)
                flat_states = traj_batch.state.reshape(batch_size, env_spec.state_dim)
                states_by_agent = jnp.broadcast_to(
                    flat_states[jnp.newaxis, :, :],
                    (num_agents, batch_size, env_spec.state_dim),
                )

                flat_batch = {
                    "obs": jnp.transpose(traj_batch.obs, (2, 0, 1, 3)).reshape(
                        num_agents, batch_size, env_spec.obs_dim
                    ),
                    "state": states_by_agent,
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
                        f"return={metrics['episode_return_mean']:.3f} "
                        f"value={metrics['value_mean']:.3f}"
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
    apply_actor: Callable[[Any, jax.Array], jax.Array],
    apply_critic: Callable[[Any, jax.Array], jax.Array],
    actor_params: Any,
    critic_params: Any,
    obs: jax.Array,
    action_mask: jax.Array,
    state: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    mask_by_agent = jnp.swapaxes(action_mask, 0, 1)
    logits = apply_actor(actor_params, obs_by_agent)
    logits = mask_logits(logits, mask_by_agent)
    values = apply_critic(critic_params, state)
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
        values,
    )


def _compute_last_values(
    apply_critic: Callable[[Any, jax.Array], jax.Array],
    critic_params: Any,
    state: jax.Array,
) -> jax.Array:
    return apply_critic(critic_params, state)


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
    actor: Actor,
    critic: CentralCritic,
    tx: optax.GradientTransformation,
    train_state: MAPPOTrainState,
    minibatch: Mapping[str, jax.Array],
) -> tuple[MAPPOTrainState, dict[str, jax.Array]]:
    clip_eps = jnp.asarray(config["CLIP_EPS"], dtype=jnp.float32)
    value_clip_eps = jnp.asarray(config["VALUE_CLIP_EPS"], dtype=jnp.float32)
    ent_coef = jnp.asarray(config["ENT_COEF"], dtype=jnp.float32)
    vf_coef = jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", True))

    def _loss_fn(
        actor_params: Any,
        critic_params: Any,
        obs: jax.Array,
        state: jax.Array,
        action: jax.Array,
        action_mask: jax.Array,
        old_log_prob: jax.Array,
        old_value: jax.Array,
        advantage: jax.Array,
        target: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = actor.apply(actor_params, obs)
        logits = mask_logits(logits, action_mask)
        value = critic.apply(critic_params, state)
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
            "value_mean": value.mean(),
        }

    grad_fn = jax.vmap(
        jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    (losses, loss_info), (actor_grads, critic_grads) = grad_fn(
        train_state.actor_params,
        train_state.critic_params,
        minibatch["obs"],
        minibatch["state"],
        minibatch["action"],
        minibatch["action_mask"],
        minibatch["log_prob"],
        minibatch["value"],
        minibatch["advantage"],
        minibatch["target"],
    )

    actor_updates, next_actor_opt_state = jax.vmap(tx.update)(
        actor_grads,
        train_state.actor_opt_state,
        train_state.actor_params,
    )
    critic_updates, next_critic_opt_state = jax.vmap(tx.update)(
        critic_grads,
        train_state.critic_opt_state,
        train_state.critic_params,
    )
    next_actor_params = jax.vmap(optax.apply_updates)(
        train_state.actor_params,
        actor_updates,
    )
    next_critic_params = jax.vmap(optax.apply_updates)(
        train_state.critic_params,
        critic_updates,
    )

    metrics = {
        key: value
        for key, value in loss_info.items()
    }
    metrics["total_loss"] = losses

    return (
        MAPPOTrainState(
            actor_params=next_actor_params,
            critic_params=next_critic_params,
            actor_opt_state=next_actor_opt_state,
            critic_opt_state=next_critic_opt_state,
        ),
        metrics,
    )

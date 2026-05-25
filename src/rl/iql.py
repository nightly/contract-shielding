from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal

from .ippo import (
    EnvFactory,
    ProgressCallback,
    assert_actions_respect_masks,
    actions_array_to_dict,
    action_masks_from_infos,
    all_true_action_masks,
    dict_observations_to_array,
    dict_values_to_array,
    masked_argmax,
    masked_max,
    resolve_env_factory,
    sample_masked_random_actions,
    shared_info_value,
    validate_parallel_env,
)
from .trajectory import episode_history_entry


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        hidden = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        hidden = activation(hidden)
        hidden = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(hidden)
        hidden = activation(hidden)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(hidden)


class IQLTrainState(NamedTuple):
    params: Any
    target_params: Any
    opt_state: Any
    grad_steps: jax.Array


class ReplayBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    next_obs: jax.Array
    next_action_mask: jax.Array


class ReplayBuffer(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    next_obs: np.ndarray
    next_action_mask: np.ndarray
    size: int
    position: int


def init_train_state(
    rng: jax.Array,
    network: QNetwork,
    tx: optax.GradientTransformation,
    *,
    num_agents: int,
    obs_dim: int,
) -> IQLTrainState:
    init_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    init_rngs = jax.random.split(rng, num_agents)
    params = jax.vmap(network.init)(init_rngs, init_obs)
    opt_state = jax.vmap(tx.init)(params)
    return IQLTrainState(
        params=params,
        target_params=params,
        opt_state=opt_state,
        grad_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def init_replay_buffer(
    *,
    capacity: int,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
) -> ReplayBuffer:
    if capacity <= 0:
        raise ValueError("IQL replay buffer capacity must be positive.")

    return ReplayBuffer(
        obs=np.zeros((capacity, num_agents, obs_dim), dtype=np.float32),
        action=np.zeros((capacity, num_agents), dtype=np.int32),
        reward=np.zeros((capacity, num_agents), dtype=np.float32),
        terminated=np.zeros((capacity, num_agents), dtype=np.float32),
        next_obs=np.zeros((capacity, num_agents, obs_dim), dtype=np.float32),
        next_action_mask=np.ones((capacity, num_agents, action_dim), dtype=bool),
        size=0,
        position=0,
    )


def add_to_replay_buffer(
    buffer: ReplayBuffer,
    *,
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    terminated: np.ndarray,
    next_obs: np.ndarray,
    next_action_mask: np.ndarray,
) -> ReplayBuffer:
    flat_obs = obs.reshape((-1, *obs.shape[-2:]))
    flat_action = action.reshape((-1, action.shape[-1]))
    flat_reward = reward.reshape((-1, reward.shape[-1]))
    flat_terminated = terminated.reshape((-1, terminated.shape[-1]))
    flat_next_obs = next_obs.reshape((-1, *next_obs.shape[-2:]))
    flat_next_action_mask = next_action_mask.reshape((-1, *next_action_mask.shape[-2:]))

    if not (
        flat_obs.shape[0]
        == flat_action.shape[0]
        == flat_reward.shape[0]
        == flat_terminated.shape[0]
        == flat_next_obs.shape[0]
        == flat_next_action_mask.shape[0]
    ):
        raise ValueError("Replay transition arrays must contain the same item count.")

    capacity = buffer.obs.shape[0]
    obs_store = buffer.obs.copy()
    action_store = buffer.action.copy()
    reward_store = buffer.reward.copy()
    terminated_store = buffer.terminated.copy()
    next_obs_store = buffer.next_obs.copy()
    next_action_mask_store = buffer.next_action_mask.copy()

    num_items = int(flat_obs.shape[0])
    if num_items >= capacity:
        obs_store[:] = flat_obs[-capacity:]
        action_store[:] = flat_action[-capacity:]
        reward_store[:] = flat_reward[-capacity:]
        terminated_store[:] = flat_terminated[-capacity:]
        next_obs_store[:] = flat_next_obs[-capacity:]
        next_action_mask_store[:] = flat_next_action_mask[-capacity:]
        return ReplayBuffer(
            obs=obs_store,
            action=action_store,
            reward=reward_store,
            terminated=terminated_store,
            next_obs=next_obs_store,
            next_action_mask=next_action_mask_store,
            size=capacity,
            position=0,
        )

    position = int(buffer.position)
    remaining = num_items
    offset = 0
    while remaining:
        end = min(capacity, position + remaining)
        chunk_size = end - position
        source = slice(offset, offset + chunk_size)
        target = slice(position, end)
        obs_store[target] = flat_obs[source]
        action_store[target] = flat_action[source]
        reward_store[target] = flat_reward[source]
        terminated_store[target] = flat_terminated[source]
        next_obs_store[target] = flat_next_obs[source]
        next_action_mask_store[target] = flat_next_action_mask[source]
        remaining -= chunk_size
        offset += chunk_size
        position = (position + chunk_size) % capacity

    return ReplayBuffer(
        obs=obs_store,
        action=action_store,
        reward=reward_store,
        terminated=terminated_store,
        next_obs=next_obs_store,
        next_action_mask=next_action_mask_store,
        size=min(capacity, int(buffer.size) + num_items),
        position=position,
    )


def sample_replay_buffer(
    buffer: ReplayBuffer,
    *,
    batch_size: int,
    rng: jax.Array,
) -> ReplayBatch:
    if buffer.size <= 0:
        raise ValueError("Cannot sample from an empty IQL replay buffer.")

    indices = np.asarray(
        jax.random.randint(rng, (int(batch_size),), 0, int(buffer.size)),
        dtype=np.int32,
    )
    return ReplayBatch(
        obs=jnp.asarray(buffer.obs[indices]),
        action=jnp.asarray(buffer.action[indices]),
        reward=jnp.asarray(buffer.reward[indices]),
        terminated=jnp.asarray(buffer.terminated[indices]),
        next_obs=jnp.asarray(buffer.next_obs[indices]),
        next_action_mask=jnp.asarray(buffer.next_action_mask[indices]),
    )


def _positive_int_config(
    config: Mapping[str, Any],
    key: str,
    *,
    algorithm_name: str,
) -> int:
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{algorithm_name} requires {key} to be positive.")
    return value


def _nonnegative_int_config(
    config: Mapping[str, Any],
    key: str,
    *,
    algorithm_name: str,
) -> int:
    value = int(config[key])
    if value < 0:
        raise ValueError(f"{algorithm_name} requires {key} to be nonnegative.")
    return value


def _compute_num_updates(
    config: Mapping[str, Any],
    *,
    algorithm_name: str,
) -> int:
    total_timesteps = _positive_int_config(
        config,
        "TOTAL_TIMESTEPS",
        algorithm_name=algorithm_name,
    )
    num_steps = _positive_int_config(
        config,
        "NUM_STEPS",
        algorithm_name=algorithm_name,
    )
    num_envs = _positive_int_config(
        config,
        "NUM_ENVS",
        algorithm_name=algorithm_name,
    )
    num_updates = total_timesteps // num_steps // num_envs
    if num_updates <= 0:
        raise ValueError(
            f"{algorithm_name} requires TOTAL_TIMESTEPS to cover at least one "
            "NUM_STEPS * NUM_ENVS rollout update."
        )
    return num_updates


def _validate_iql_config(
    config: Mapping[str, Any],
    *,
    algorithm_name: str,
) -> None:
    _positive_int_config(
        config,
        "IQL_BUFFER_SIZE",
        algorithm_name=algorithm_name,
    )
    _positive_int_config(
        config,
        "IQL_BUFFER_BATCH_SIZE",
        algorithm_name=algorithm_name,
    )
    _positive_int_config(
        config,
        "IQL_UPDATE_EPOCHS",
        algorithm_name=algorithm_name,
    )
    _positive_int_config(
        config,
        "IQL_TARGET_UPDATE_INTERVAL",
        algorithm_name=algorithm_name,
    )
    _nonnegative_int_config(
        config,
        "IQL_LEARNING_STARTS",
        algorithm_name=algorithm_name,
    )


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    algorithm_name = "IQL"
    config = dict(config)
    config.setdefault("IQL_HIDDEN_SIZE", 64)
    config.setdefault("IQL_BUFFER_SIZE", 50_000)
    config.setdefault(
        "IQL_BUFFER_BATCH_SIZE",
        int(config["NUM_ENVS"]) * int(config["NUM_STEPS"]),
    )
    config.setdefault(
        "IQL_LEARNING_STARTS",
        int(config["NUM_ENVS"]) * int(config["NUM_STEPS"]),
    )
    config.setdefault("IQL_UPDATE_EPOCHS", config["UPDATE_EPOCHS"])
    config.setdefault("IQL_TARGET_UPDATE_INTERVAL", 1)
    config.setdefault("IQL_TAU", 1.0)
    config.setdefault("IQL_EPS_START", 1.0)
    config.setdefault("IQL_EPS_FINISH", 0.05)
    config.setdefault("IQL_EPS_DECAY", 0.8)
    config.setdefault("IQL_DOUBLE_Q", True)
    config.setdefault("IQL_ANNEAL_LR", bool(config.get("ANNEAL_LR", False)))
    config["NUM_UPDATES"] = _compute_num_updates(
        config,
        algorithm_name=algorithm_name,
    )
    _validate_iql_config(config, algorithm_name=algorithm_name)

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

    network = QNetwork(
        action_dim=env_spec.action_dim,
        hidden_size=int(config["IQL_HIDDEN_SIZE"]),
        activation=config.get("ACTIVATION", "tanh"),
    )

    total_grad_steps = max(int(config["NUM_UPDATES"]) * int(config["IQL_UPDATE_EPOCHS"]), 1)

    def linear_schedule(step_count: jax.Array) -> jax.Array:
        frac = 1.0 - step_count / total_grad_steps
        return jnp.asarray(config["LR"], dtype=jnp.float32) * frac

    if config.get("IQL_ANNEAL_LR", False):
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    eps_scheduler = optax.linear_schedule(
        init_value=float(config["IQL_EPS_START"]),
        end_value=float(config["IQL_EPS_FINISH"]),
        transition_steps=max(
            int(float(config["IQL_EPS_DECAY"]) * max(int(config["NUM_UPDATES"]), 1)),
            1,
        ),
    )

    def apply_q(params: Any, obs: jax.Array) -> jax.Array:
        return jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(params, obs)

    sample_actions_jit = jax.jit(
        lambda params, obs, action_mask, epsilon, rng: _sample_actions(
            apply_q,
            params,
            obs,
            action_mask,
            epsilon,
            rng,
            env_spec.action_dim,
        )
    )
    update_minibatch_jit = jax.jit(
        lambda state, batch: _update_minibatch(config, network, tx, state, batch)
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        rng = jax.random.PRNGKey(int(rng)) if np.isscalar(rng) else rng
        num_envs = int(config["NUM_ENVS"])
        num_steps = int(config["NUM_STEPS"])
        num_agents = len(env_spec.agent_ids)
        batch_size = num_steps * num_envs
        replay_batch_size = int(config["IQL_BUFFER_BATCH_SIZE"])
        target_update_interval = max(int(config["IQL_TARGET_UPDATE_INTERVAL"]), 1)

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
            replay_buffer = init_replay_buffer(
                capacity=int(config["IQL_BUFFER_SIZE"]),
                num_agents=num_agents,
                obs_dim=env_spec.obs_dim,
                action_dim=env_spec.action_dim,
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
                epsilon = float(eps_scheduler(update_idx))
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_next_obs = np.zeros_like(traj_obs)
                traj_next_action_masks = np.ones(
                    (num_steps, num_envs, num_agents, env_spec.action_dim),
                    dtype=bool,
                )
                traj_actions = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.int32,
                )
                traj_action_q_values = np.zeros(
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
                completed_returns: list[np.ndarray] = []
                completed_lengths: list[int] = []
                completed_safety_violation_counts: list[float] = []

                for step_idx in range(num_steps):
                    obs_batch = jnp.asarray(last_obs)
                    action_mask_batch = jnp.asarray(last_action_masks)
                    rng, sample_rng = jax.random.split(rng)
                    actions, action_q_values = sample_actions_jit(
                        train_state.params,
                        obs_batch,
                        action_mask_batch,
                        jnp.asarray(epsilon, dtype=jnp.float32),
                        sample_rng,
                    )
                    actions_np = np.asarray(actions, dtype=np.int32)
                    action_q_values_np = np.asarray(action_q_values, dtype=np.float32)

                    next_obs = np.zeros_like(last_obs)
                    next_obs_for_target = np.zeros_like(last_obs)
                    next_action_masks_for_target = all_true_action_masks(
                        num_envs,
                        num_agents,
                        env_spec.action_dim,
                    )
                    reward_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    terminated_batch = np.zeros((num_envs, num_agents), dtype=np.float32)

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
                            phase="IQL rollout sampling",
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
                        step_action_masks = action_masks_from_infos(
                            infos,
                            env_spec.agent_ids,
                            env_spec.action_dim,
                        )

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
                        if not np.all(done_array == done_array[0]):
                            raise ValueError(
                                "IQL expects all agents in an environment to finish together."
                            )

                        if all(agent in obs_dict for agent in env_spec.agent_ids):
                            next_obs_for_target[env_idx] = dict_observations_to_array(
                                obs_dict,
                                env_spec.agent_ids,
                                env_spec.observation_spaces,
                            )
                            next_action_masks_for_target[env_idx] = step_action_masks

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

                        reward_batch[env_idx] = reward_array
                        terminated_batch[env_idx] = terminated_array
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
                    traj_next_obs[step_idx] = next_obs_for_target
                    traj_next_action_masks[step_idx] = next_action_masks_for_target
                    traj_actions[step_idx] = actions_np
                    traj_action_q_values[step_idx] = action_q_values_np
                    traj_rewards[step_idx] = reward_batch
                    traj_terminated[step_idx] = terminated_batch
                    last_obs = next_obs

                replay_buffer = add_to_replay_buffer(
                    replay_buffer,
                    obs=traj_obs,
                    action=traj_actions,
                    reward=traj_rewards,
                    terminated=traj_terminated,
                    next_obs=traj_next_obs,
                    next_action_mask=traj_next_action_masks,
                )

                can_learn = (
                    replay_buffer.size >= replay_batch_size
                    and (update_idx + 1) * batch_size >= int(config["IQL_LEARNING_STARTS"])
                )
                minibatch_metrics: list[dict[str, np.ndarray]] = []
                if can_learn:
                    for _ in range(int(config["IQL_UPDATE_EPOCHS"])):
                        rng, sample_rng = jax.random.split(rng)
                        minibatch = sample_replay_buffer(
                            replay_buffer,
                            batch_size=replay_batch_size,
                            rng=sample_rng,
                        )
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

                if (update_idx + 1) % target_update_interval == 0:
                    train_state = IQLTrainState(
                        params=train_state.params,
                        target_params=optax.incremental_update(
                            train_state.params,
                            train_state.target_params,
                            float(config["IQL_TAU"]),
                        ),
                        opt_state=train_state.opt_state,
                        grad_steps=train_state.grad_steps,
                    )

                if minibatch_metrics:
                    aggregated_loss_metrics = {
                        key: np.stack(
                            [metric[key] for metric in minibatch_metrics],
                            axis=0,
                        ).mean(axis=0)
                        for key in minibatch_metrics[0]
                    }
                else:
                    rollout_q_values = traj_action_q_values.mean(axis=(0, 1))
                    aggregated_loss_metrics = {
                        "q_loss": np.zeros(num_agents, dtype=np.float32),
                        "td_error_mean": np.zeros(num_agents, dtype=np.float32),
                        "q_value_mean": rollout_q_values,
                    }

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
                    "epsilon": epsilon,
                    "grad_steps": float(np.asarray(train_state.grad_steps)),
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
                    metrics[f"{agent_id}/epsilon"] = epsilon
                    for key, value in aggregated_loss_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f} "
                        f"eps={metrics['epsilon']:.3f} "
                        f"q_loss={metrics['q_loss']:.5f}"
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
    apply_q: Callable[[Any, jax.Array], jax.Array],
    params: Any,
    obs: jax.Array,
    action_mask: jax.Array,
    epsilon: jax.Array,
    rng: jax.Array,
    action_dim: int,
) -> tuple[jax.Array, jax.Array]:
    del action_dim
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    mask_by_agent = jnp.swapaxes(action_mask, 0, 1)
    q_values_by_agent = apply_q(params, obs_by_agent)
    q_values = jnp.swapaxes(q_values_by_agent, 0, 1)
    greedy_actions = masked_argmax(q_values, action_mask)

    rng_random, rng_explore = jax.random.split(rng)
    random_actions_by_agent = jax.vmap(
        lambda mask, agent_rng: sample_masked_random_actions(mask, agent_rng),
    )(mask_by_agent, jax.random.split(rng_random, mask_by_agent.shape[0]))
    random_actions = jnp.swapaxes(random_actions_by_agent, 0, 1)
    explore = jax.random.uniform(rng_explore, greedy_actions.shape) < epsilon
    actions = jnp.where(explore, random_actions, greedy_actions)
    chosen_q_values = jnp.take_along_axis(
        q_values,
        actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    return actions, chosen_q_values


def _compute_td_targets(
    reward: jax.Array,
    terminated: jax.Array,
    next_q: jax.Array,
    *,
    gamma: float | jax.Array,
) -> jax.Array:
    return reward + (1.0 - terminated) * jnp.asarray(gamma, dtype=jnp.float32) * next_q


def _update_minibatch(
    config: Mapping[str, Any],
    network: QNetwork,
    tx: optax.GradientTransformation,
    train_state: IQLTrainState,
    minibatch: ReplayBatch,
) -> tuple[IQLTrainState, dict[str, jax.Array]]:
    obs = jnp.swapaxes(minibatch.obs, 0, 1)
    action = jnp.swapaxes(minibatch.action, 0, 1)
    reward = jnp.swapaxes(minibatch.reward, 0, 1)
    terminated = jnp.swapaxes(minibatch.terminated, 0, 1)
    next_obs = jnp.swapaxes(minibatch.next_obs, 0, 1)
    next_action_mask = jnp.swapaxes(minibatch.next_action_mask, 0, 1)

    target_q_values = jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(
        train_state.target_params,
        next_obs,
    )
    if bool(config.get("IQL_DOUBLE_Q", True)):
        online_next_q_values = jax.vmap(
            lambda p, o: network.apply(p, o),
            in_axes=(0, 0),
        )(
            train_state.params,
            next_obs,
        )
        next_actions = masked_argmax(online_next_q_values, next_action_mask)
        next_q = jnp.take_along_axis(
            target_q_values,
            next_actions[..., jnp.newaxis],
            axis=-1,
        ).squeeze(axis=-1)
    else:
        next_q = masked_max(target_q_values, next_action_mask)

    td_targets = _compute_td_targets(
        reward,
        terminated,
        next_q,
        gamma=jnp.asarray(config["GAMMA"], dtype=jnp.float32),
    )

    def _loss_fn(
        params: Any,
        agent_obs: jax.Array,
        agent_action: jax.Array,
        agent_target: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        q_values = network.apply(params, agent_obs)
        chosen_q = jnp.take_along_axis(
            q_values,
            agent_action[..., jnp.newaxis],
            axis=-1,
        ).squeeze(axis=-1)
        td_error = chosen_q - jax.lax.stop_gradient(agent_target)
        q_loss = jnp.mean(jnp.square(td_error))
        return q_loss, {
            "q_loss": q_loss,
            "td_error_mean": jnp.mean(jnp.abs(td_error)),
            "q_value_mean": jnp.mean(chosen_q),
        }

    grad_fn = jax.vmap(
        jax.value_and_grad(_loss_fn, has_aux=True),
        in_axes=(0, 0, 0, 0),
    )
    (losses, loss_info), grads = grad_fn(
        train_state.params,
        obs,
        action,
        td_targets,
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
    metrics["q_loss"] = losses

    return (
        IQLTrainState(
            params=next_params,
            target_params=train_state.target_params,
            opt_state=next_opt_state,
            grad_steps=train_state.grad_steps + 1,
        ),
        metrics,
    )


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 64,
        "UPDATE_EPOCHS": 2,
        "GAMMA": 0.99,
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

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .iql import (
    QNetwork,
    _compute_num_updates,
    _compute_td_targets,
    _validate_iql_config,
)
from .ippo import (
    EnvFactory,
    ProgressCallback,
    actions_array_to_dict,
    dict_infos_to_array,
    dict_observations_to_array,
    dict_values_to_array,
    resolve_env_factory,
    shared_info_value,
    validate_parallel_env,
)
from .trajectory import discounted_cost_returns_at_starts, episode_history_entry


class IQLLagrangianTrainState(NamedTuple):
    reward_params: Any
    reward_target_params: Any
    reward_opt_state: Any
    cost_params: Any
    cost_target_params: Any
    cost_opt_state: Any
    lagrangian_multipliers: jax.Array
    grad_steps: jax.Array


class CostReplayBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    cost: jax.Array
    terminated: jax.Array
    next_obs: jax.Array


class CostReplayBuffer(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    cost: np.ndarray
    terminated: np.ndarray
    next_obs: np.ndarray
    size: int
    position: int


def init_train_state(
    rng: jax.Array,
    network: QNetwork,
    tx: optax.GradientTransformation,
    *,
    num_agents: int,
    obs_dim: int,
    lambda_init: float,
) -> IQLLagrangianTrainState:
    reward_rng, cost_rng = jax.random.split(rng)
    init_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    reward_rngs = jax.random.split(reward_rng, num_agents)
    cost_rngs = jax.random.split(cost_rng, num_agents)
    reward_params = jax.vmap(network.init)(reward_rngs, init_obs)
    cost_params = jax.vmap(network.init)(cost_rngs, init_obs)
    return IQLLagrangianTrainState(
        reward_params=reward_params,
        reward_target_params=reward_params,
        reward_opt_state=jax.vmap(tx.init)(reward_params),
        cost_params=cost_params,
        cost_target_params=cost_params,
        cost_opt_state=jax.vmap(tx.init)(cost_params),
        lagrangian_multipliers=jnp.full(
            (num_agents,),
            float(lambda_init),
            dtype=jnp.float32,
        ),
        grad_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def init_replay_buffer(
    *,
    capacity: int,
    num_agents: int,
    obs_dim: int,
) -> CostReplayBuffer:
    if capacity <= 0:
        raise ValueError("IQL-Lagrangian replay buffer capacity must be positive.")

    return CostReplayBuffer(
        obs=np.zeros((capacity, num_agents, obs_dim), dtype=np.float32),
        action=np.zeros((capacity, num_agents), dtype=np.int32),
        reward=np.zeros((capacity, num_agents), dtype=np.float32),
        cost=np.zeros((capacity, num_agents), dtype=np.float32),
        terminated=np.zeros((capacity, num_agents), dtype=np.float32),
        next_obs=np.zeros((capacity, num_agents, obs_dim), dtype=np.float32),
        size=0,
        position=0,
    )


def add_to_replay_buffer(
    buffer: CostReplayBuffer,
    *,
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    cost: np.ndarray,
    terminated: np.ndarray,
    next_obs: np.ndarray,
) -> CostReplayBuffer:
    flat_obs = obs.reshape((-1, *obs.shape[-2:]))
    flat_action = action.reshape((-1, action.shape[-1]))
    flat_reward = reward.reshape((-1, reward.shape[-1]))
    flat_cost = cost.reshape((-1, cost.shape[-1]))
    flat_terminated = terminated.reshape((-1, terminated.shape[-1]))
    flat_next_obs = next_obs.reshape((-1, *next_obs.shape[-2:]))

    if not (
        flat_obs.shape[0]
        == flat_action.shape[0]
        == flat_reward.shape[0]
        == flat_cost.shape[0]
        == flat_terminated.shape[0]
        == flat_next_obs.shape[0]
    ):
        raise ValueError("Replay transition arrays must contain the same item count.")

    capacity = buffer.obs.shape[0]
    obs_store = buffer.obs.copy()
    action_store = buffer.action.copy()
    reward_store = buffer.reward.copy()
    cost_store = buffer.cost.copy()
    terminated_store = buffer.terminated.copy()
    next_obs_store = buffer.next_obs.copy()

    num_items = int(flat_obs.shape[0])
    if num_items >= capacity:
        obs_store[:] = flat_obs[-capacity:]
        action_store[:] = flat_action[-capacity:]
        reward_store[:] = flat_reward[-capacity:]
        cost_store[:] = flat_cost[-capacity:]
        terminated_store[:] = flat_terminated[-capacity:]
        next_obs_store[:] = flat_next_obs[-capacity:]
        return CostReplayBuffer(
            obs=obs_store,
            action=action_store,
            reward=reward_store,
            cost=cost_store,
            terminated=terminated_store,
            next_obs=next_obs_store,
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
        cost_store[target] = flat_cost[source]
        terminated_store[target] = flat_terminated[source]
        next_obs_store[target] = flat_next_obs[source]
        remaining -= chunk_size
        offset += chunk_size
        position = (position + chunk_size) % capacity

    return CostReplayBuffer(
        obs=obs_store,
        action=action_store,
        reward=reward_store,
        cost=cost_store,
        terminated=terminated_store,
        next_obs=next_obs_store,
        size=min(capacity, int(buffer.size) + num_items),
        position=position,
    )


def sample_replay_buffer(
    buffer: CostReplayBuffer,
    *,
    batch_size: int,
    rng: jax.Array,
) -> CostReplayBatch:
    if buffer.size <= 0:
        raise ValueError("Cannot sample from an empty IQL-Lagrangian replay buffer.")

    indices = np.asarray(
        jax.random.randint(rng, (int(batch_size),), 0, int(buffer.size)),
        dtype=np.int32,
    )
    return CostReplayBatch(
        obs=jnp.asarray(buffer.obs[indices]),
        action=jnp.asarray(buffer.action[indices]),
        reward=jnp.asarray(buffer.reward[indices]),
        cost=jnp.asarray(buffer.cost[indices]),
        terminated=jnp.asarray(buffer.terminated[indices]),
        next_obs=jnp.asarray(buffer.next_obs[indices]),
    )


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    algorithm_name = "IQL-Lagrangian"
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
    config.setdefault("COST_INFO_KEY", "safety_violation")
    config.setdefault("COST_LIMIT", 0.0)
    config.setdefault("LAMBDA_LR", 0.05)
    config.setdefault("LAMBDA_INIT", 0.0)
    config.setdefault("COST_GAMMA", config["GAMMA"])
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
        lambda reward_params, cost_params, lambdas, obs, epsilon, rng: _sample_actions(
            apply_q,
            reward_params,
            cost_params,
            lambdas,
            obs,
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
        episode_costs = np.zeros((num_envs, num_agents), dtype=np.float32)
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
                lambda_init=float(config["LAMBDA_INIT"]),
            )
            replay_buffer = init_replay_buffer(
                capacity=int(config["IQL_BUFFER_SIZE"]),
                num_agents=num_agents,
                obs_dim=env_spec.obs_dim,
            )

            last_obs = np.zeros(
                (num_envs, num_agents, env_spec.obs_dim),
                dtype=np.float32,
            )
            for env_idx, env in enumerate(envs):
                obs_dict, _ = env.reset(seed=int(config.get("SEED", 0)) + reset_counter)
                reset_counter += 1
                last_obs[env_idx] = dict_observations_to_array(
                    obs_dict,
                    env_spec.agent_ids,
                    env_spec.observation_spaces,
                )

            for update_idx in range(int(config["NUM_UPDATES"])):
                epsilon = float(eps_scheduler(update_idx))
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_next_obs = np.zeros_like(traj_obs)
                traj_actions = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.int32,
                )
                traj_reward_q_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_cost_q_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_rewards = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_costs = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_episode_dones = np.zeros(
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
                completed_cost_returns: list[np.ndarray] = []
                completed_lengths: list[int] = []
                completed_safety_violation_counts: list[float] = []

                for step_idx in range(num_steps):
                    obs_batch = jnp.asarray(last_obs)
                    rng, sample_rng = jax.random.split(rng)
                    actions, reward_q_values, cost_q_values = sample_actions_jit(
                        train_state.reward_params,
                        train_state.cost_params,
                        train_state.lagrangian_multipliers,
                        obs_batch,
                        jnp.asarray(epsilon, dtype=jnp.float32),
                        sample_rng,
                    )
                    actions_np = np.asarray(actions, dtype=np.int32)
                    reward_q_values_np = np.asarray(reward_q_values, dtype=np.float32)
                    cost_q_values_np = np.asarray(cost_q_values, dtype=np.float32)

                    next_obs = np.zeros_like(last_obs)
                    next_obs_for_target = np.zeros_like(last_obs)
                    reward_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    cost_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    terminated_batch = np.zeros((num_envs, num_agents), dtype=np.float32)

                    for env_idx, env in enumerate(envs):
                        env_global_step = (
                            update_idx * batch_size
                            + step_idx * num_envs
                            + env_idx
                            + 1
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
                        cost_array = dict_infos_to_array(
                            infos,
                            env_spec.agent_ids,
                            str(config["COST_INFO_KEY"]),
                            default=0.0,
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
                                "IQL-Lagrangian expects all agents in an environment "
                                "to finish together."
                            )

                        if all(agent in obs_dict for agent in env_spec.agent_ids):
                            next_obs_for_target[env_idx] = dict_observations_to_array(
                                obs_dict,
                                env_spec.agent_ids,
                                env_spec.observation_spaces,
                            )

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
                        cost_batch[env_idx] = cost_array
                        terminated_batch[env_idx] = terminated_array
                        traj_episode_dones[step_idx, env_idx] = done_array
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
                        episode_costs[env_idx] += cost_array
                        episode_lengths[env_idx] += 1

                        if bool(done_array[0]):
                            completed_returns.append(episode_returns[env_idx].copy())
                            completed_cost_returns.append(episode_costs[env_idx].copy())
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
                            obs_dict, _ = env.reset(
                                seed=int(config.get("SEED", 0)) + reset_counter
                            )
                            reset_counter += 1
                            episode_returns[env_idx].fill(0.0)
                            episode_costs[env_idx].fill(0.0)
                            episode_lengths[env_idx] = 0
                            episode_start_steps[env_idx] = env_global_step

                        next_obs[env_idx] = dict_observations_to_array(
                            obs_dict,
                            env_spec.agent_ids,
                            env_spec.observation_spaces,
                        )

                    traj_obs[step_idx] = last_obs
                    traj_next_obs[step_idx] = next_obs_for_target
                    traj_actions[step_idx] = actions_np
                    traj_reward_q_values[step_idx] = reward_q_values_np
                    traj_cost_q_values[step_idx] = cost_q_values_np
                    traj_rewards[step_idx] = reward_batch
                    traj_costs[step_idx] = cost_batch
                    traj_terminated[step_idx] = terminated_batch
                    last_obs = next_obs

                replay_buffer = add_to_replay_buffer(
                    replay_buffer,
                    obs=traj_obs,
                    action=traj_actions,
                    reward=traj_rewards,
                    cost=traj_costs,
                    terminated=traj_terminated,
                    next_obs=traj_next_obs,
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

                cost_returns = discounted_cost_returns_at_starts(
                    traj_costs,
                    traj_episode_dones,
                    gamma=float(config["COST_GAMMA"]),
                )
                next_lambdas = jnp.maximum(
                    0.0,
                    train_state.lagrangian_multipliers
                    + jnp.asarray(config["LAMBDA_LR"], dtype=jnp.float32)
                    * (
                        jnp.asarray(cost_returns, dtype=jnp.float32)
                        - jnp.asarray(config["COST_LIMIT"], dtype=jnp.float32)
                    ),
                )
                train_state = IQLLagrangianTrainState(
                    reward_params=train_state.reward_params,
                    reward_target_params=train_state.reward_target_params,
                    reward_opt_state=train_state.reward_opt_state,
                    cost_params=train_state.cost_params,
                    cost_target_params=train_state.cost_target_params,
                    cost_opt_state=train_state.cost_opt_state,
                    lagrangian_multipliers=next_lambdas,
                    grad_steps=train_state.grad_steps,
                )

                if (update_idx + 1) % target_update_interval == 0:
                    train_state = IQLLagrangianTrainState(
                        reward_params=train_state.reward_params,
                        reward_target_params=optax.incremental_update(
                            train_state.reward_params,
                            train_state.reward_target_params,
                            float(config["IQL_TAU"]),
                        ),
                        reward_opt_state=train_state.reward_opt_state,
                        cost_params=train_state.cost_params,
                        cost_target_params=optax.incremental_update(
                            train_state.cost_params,
                            train_state.cost_target_params,
                            float(config["IQL_TAU"]),
                        ),
                        cost_opt_state=train_state.cost_opt_state,
                        lagrangian_multipliers=train_state.lagrangian_multipliers,
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
                    aggregated_loss_metrics = {
                        "reward_q_loss": np.zeros(num_agents, dtype=np.float32),
                        "cost_q_loss": np.zeros(num_agents, dtype=np.float32),
                        "reward_td_error_mean": np.zeros(num_agents, dtype=np.float32),
                        "cost_td_error_mean": np.zeros(num_agents, dtype=np.float32),
                        "reward_q_value_mean": traj_reward_q_values.mean(axis=(0, 1)),
                        "cost_q_value_mean": traj_cost_q_values.mean(axis=(0, 1)),
                    }

                step_reward_mean = traj_rewards.mean(axis=(0, 1))
                step_cost_mean = traj_costs.mean(axis=(0, 1))
                step_cost_rate = (traj_costs > 0.0).mean(axis=(0, 1))
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
                    mean_episode_cost_return = np.stack(
                        completed_cost_returns,
                        axis=0,
                    ).mean(axis=0)
                    mean_episode_length = float(np.mean(completed_lengths))
                    mean_episode_safety_violations = float(
                        np.mean(completed_safety_violation_counts)
                    )
                else:
                    mean_episode_return = np.zeros(num_agents, dtype=np.float32)
                    mean_episode_cost_return = np.zeros(num_agents, dtype=np.float32)
                    mean_episode_length = 0.0
                    mean_episode_safety_violations = 0.0

                lambda_values = np.asarray(train_state.lagrangian_multipliers)
                metrics = {
                    "update": float(update_idx),
                    "global_step": float((update_idx + 1) * batch_size),
                    "completed_episodes": float(total_completed_episodes),
                    "episode_length": mean_episode_length,
                    "reward_mean": float(step_reward_mean.mean()),
                    "episode_return_mean": float(mean_episode_return.mean()),
                    "cost_mean": float(step_cost_mean.mean()),
                    "cost_return_mean": float(mean_episode_cost_return.mean()),
                    "cost_rate_mean": float(step_cost_rate.mean()),
                    "lagrangian_multiplier_mean": float(lambda_values.mean()),
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
                    metrics[f"{agent_id}/cost_mean"] = float(step_cost_mean[agent_idx])
                    metrics[f"{agent_id}/cost_return"] = float(
                        mean_episode_cost_return[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_rate"] = float(step_cost_rate[agent_idx])
                    metrics[f"{agent_id}/lagrangian_multiplier"] = float(
                        lambda_values[agent_idx]
                    )
                    metrics[f"{agent_id}/epsilon"] = epsilon
                    for key, value in aggregated_loss_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f} "
                        f"cost={metrics['cost_return_mean']:.3f} "
                        f"lambda={metrics['lagrangian_multiplier_mean']:.3f}"
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
    reward_params: Any,
    cost_params: Any,
    lagrangian_multipliers: jax.Array,
    obs: jax.Array,
    epsilon: jax.Array,
    rng: jax.Array,
    action_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    reward_q_by_agent = apply_q(reward_params, obs_by_agent)
    cost_q_by_agent = apply_q(cost_params, obs_by_agent)
    scores_by_agent = reward_q_by_agent - (
        lagrangian_multipliers[:, jnp.newaxis, jnp.newaxis] * cost_q_by_agent
    )
    scores = jnp.swapaxes(scores_by_agent, 0, 1)
    reward_q_values = jnp.swapaxes(reward_q_by_agent, 0, 1)
    cost_q_values = jnp.swapaxes(cost_q_by_agent, 0, 1)
    greedy_actions = jnp.argmax(scores, axis=-1)

    rng_random, rng_explore = jax.random.split(rng)
    random_actions = jax.random.randint(
        rng_random,
        greedy_actions.shape,
        minval=0,
        maxval=action_dim,
    )
    explore = jax.random.uniform(rng_explore, greedy_actions.shape) < epsilon
    actions = jnp.where(explore, random_actions, greedy_actions)
    chosen_reward_q_values = jnp.take_along_axis(
        reward_q_values,
        actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    chosen_cost_q_values = jnp.take_along_axis(
        cost_q_values,
        actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    return actions, chosen_reward_q_values, chosen_cost_q_values


def _update_minibatch(
    config: Mapping[str, Any],
    network: QNetwork,
    tx: optax.GradientTransformation,
    train_state: IQLLagrangianTrainState,
    minibatch: CostReplayBatch,
) -> tuple[IQLLagrangianTrainState, dict[str, jax.Array]]:
    obs = jnp.swapaxes(minibatch.obs, 0, 1)
    action = jnp.swapaxes(minibatch.action, 0, 1)
    reward = jnp.swapaxes(minibatch.reward, 0, 1)
    cost = jnp.swapaxes(minibatch.cost, 0, 1)
    terminated = jnp.swapaxes(minibatch.terminated, 0, 1)
    next_obs = jnp.swapaxes(minibatch.next_obs, 0, 1)

    reward_target_q = jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(
        train_state.reward_target_params,
        next_obs,
    )
    cost_target_q = jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(
        train_state.cost_target_params,
        next_obs,
    )
    if bool(config.get("IQL_DOUBLE_Q", True)):
        online_reward_next_q = jax.vmap(
            lambda p, o: network.apply(p, o),
            in_axes=(0, 0),
        )(
            train_state.reward_params,
            next_obs,
        )
        online_cost_next_q = jax.vmap(
            lambda p, o: network.apply(p, o),
            in_axes=(0, 0),
        )(
            train_state.cost_params,
            next_obs,
        )
        next_scores = online_reward_next_q - (
            train_state.lagrangian_multipliers[:, jnp.newaxis, jnp.newaxis]
            * online_cost_next_q
        )
        next_actions = jnp.argmax(next_scores, axis=-1)
    else:
        target_scores = reward_target_q - (
            train_state.lagrangian_multipliers[:, jnp.newaxis, jnp.newaxis]
            * cost_target_q
        )
        next_actions = jnp.argmax(target_scores, axis=-1)

    next_reward_q = jnp.take_along_axis(
        reward_target_q,
        next_actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    next_cost_q = jnp.take_along_axis(
        cost_target_q,
        next_actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    reward_targets = _compute_td_targets(
        reward,
        terminated,
        next_reward_q,
        gamma=jnp.asarray(config["GAMMA"], dtype=jnp.float32),
    )
    cost_targets = _compute_td_targets(
        cost,
        terminated,
        next_cost_q,
        gamma=jnp.asarray(config["COST_GAMMA"], dtype=jnp.float32),
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
            "td_error_mean": jnp.mean(jnp.abs(td_error)),
            "q_value_mean": jnp.mean(chosen_q),
        }

    grad_fn = jax.vmap(
        jax.value_and_grad(_loss_fn, has_aux=True),
        in_axes=(0, 0, 0, 0),
    )
    (reward_losses, reward_info), reward_grads = grad_fn(
        train_state.reward_params,
        obs,
        action,
        reward_targets,
    )
    (cost_losses, cost_info), cost_grads = grad_fn(
        train_state.cost_params,
        obs,
        action,
        cost_targets,
    )
    reward_updates, next_reward_opt_state = jax.vmap(tx.update)(
        reward_grads,
        train_state.reward_opt_state,
        train_state.reward_params,
    )
    cost_updates, next_cost_opt_state = jax.vmap(tx.update)(
        cost_grads,
        train_state.cost_opt_state,
        train_state.cost_params,
    )
    next_reward_params = jax.vmap(optax.apply_updates)(
        train_state.reward_params,
        reward_updates,
    )
    next_cost_params = jax.vmap(optax.apply_updates)(
        train_state.cost_params,
        cost_updates,
    )

    metrics = {
        "reward_q_loss": reward_losses,
        "cost_q_loss": cost_losses,
        "reward_td_error_mean": reward_info["td_error_mean"],
        "cost_td_error_mean": cost_info["td_error_mean"],
        "reward_q_value_mean": reward_info["q_value_mean"],
        "cost_q_value_mean": cost_info["q_value_mean"],
    }

    return (
        IQLLagrangianTrainState(
            reward_params=next_reward_params,
            reward_target_params=train_state.reward_target_params,
            reward_opt_state=next_reward_opt_state,
            cost_params=next_cost_params,
            cost_target_params=train_state.cost_target_params,
            cost_opt_state=next_cost_opt_state,
            lagrangian_multipliers=train_state.lagrangian_multipliers,
            grad_steps=train_state.grad_steps + 1,
        ),
        metrics,
    )

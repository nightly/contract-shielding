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


class PQNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 256
    num_layers: int = 2
    norm_type: str = "layer_norm"
    activation: str = "relu"
    dueling: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        norm_name = str(self.norm_type).lower()

        for _ in range(int(self.num_layers)):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            if norm_name == "layer_norm":
                x = nn.LayerNorm()(x)
            elif norm_name not in {"none", "identity", ""}:
                raise ValueError(
                    "PQN-VDN supports PQN_NORM_TYPE='layer_norm' or 'none', "
                    f"got {self.norm_type!r}."
                )
            x = activation(x)

        if self.dueling:
            advantage = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(x)
            value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(x)
            return value + advantage - jnp.mean(advantage, axis=-1, keepdims=True)

        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)


class PQNVDNTrainState(NamedTuple):
    params: Any
    opt_state: Any
    grad_steps: jax.Array


class PQNVDNMinibatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    target: jax.Array


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


def _validate_pqn_config(
    config: Mapping[str, Any],
    *,
    algorithm_name: str,
) -> None:
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
    num_minibatches = _positive_int_config(
        config,
        "PQN_NUM_MINIBATCHES",
        algorithm_name=algorithm_name,
    )
    _positive_int_config(
        config,
        "PQN_UPDATE_EPOCHS",
        algorithm_name=algorithm_name,
    )
    rollout_batch_size = num_steps * num_envs
    if rollout_batch_size % num_minibatches != 0:
        raise ValueError(
            "PQN-VDN requires NUM_STEPS * NUM_ENVS to be divisible by "
            "PQN_NUM_MINIBATCHES."
        )

    trace_lambda = float(config["PQN_LAMBDA"])
    if not 0.0 <= trace_lambda <= 1.0:
        raise ValueError("PQN-VDN requires PQN_LAMBDA to be in [0, 1].")

    team_reward = str(config["PQN_TEAM_REWARD"]).lower()
    if team_reward not in {"sum", "mean"}:
        raise ValueError("PQN_TEAM_REWARD must be either 'sum' or 'mean'.")


def init_train_state(
    rng: jax.Array,
    network: PQNQNetwork,
    tx: optax.GradientTransformation,
    *,
    input_dim: int,
) -> PQNVDNTrainState:
    init_obs = jnp.zeros((1, input_dim), dtype=jnp.float32)
    params = network.init(rng, init_obs)
    opt_state = tx.init(params)
    return PQNVDNTrainState(
        params=params,
        opt_state=opt_state,
        grad_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def _append_agent_ids(obs: jax.Array, *, num_agents: int) -> jax.Array:
    agent_ids = jnp.eye(num_agents, dtype=obs.dtype)
    broadcast_shape = (1,) * (obs.ndim - 2) + agent_ids.shape
    agent_ids = jnp.reshape(agent_ids, broadcast_shape)
    agent_ids = jnp.broadcast_to(agent_ids, obs.shape[:-1] + (num_agents,))
    return jnp.concatenate((obs, agent_ids), axis=-1)


def _team_reward(rewards: jax.Array, *, mode: str) -> jax.Array:
    if str(mode).lower() == "sum":
        return jnp.sum(rewards, axis=-1)
    if str(mode).lower() == "mean":
        return jnp.mean(rewards, axis=-1)
    raise ValueError("PQN_TEAM_REWARD must be either 'sum' or 'mean'.")


def _compute_q_lambda_targets(
    reward: jax.Array,
    terminated: jax.Array,
    next_team_q: jax.Array,
    *,
    gamma: float | jax.Array,
    trace_lambda: float | jax.Array,
) -> jax.Array:
    reward = jnp.asarray(reward, dtype=jnp.float32)
    terminated = jnp.asarray(terminated, dtype=jnp.float32)
    next_team_q = jnp.asarray(next_team_q, dtype=jnp.float32)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    trace_lambda = jnp.asarray(trace_lambda, dtype=jnp.float32)

    def _scan(carry: jax.Array, transition: tuple[jax.Array, jax.Array, jax.Array]):
        rew, term, next_q = transition
        target = rew + (1.0 - term) * gamma * (
            (1.0 - trace_lambda) * next_q + trace_lambda * carry
        )
        return target, target

    last_target = reward[-1] + (1.0 - terminated[-1]) * gamma * next_team_q[-1]
    if reward.shape[0] == 1:
        return last_target[jnp.newaxis]

    _, reversed_targets = jax.lax.scan(
        _scan,
        last_target,
        (reward[:-1], terminated[:-1], next_team_q[:-1]),
        reverse=True,
    )
    return jnp.concatenate((reversed_targets, last_target[jnp.newaxis]), axis=0)


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    algorithm_name = "PQN-VDN"
    config = dict(config)
    config.setdefault("PQN_HIDDEN_SIZE", 256)
    config.setdefault("PQN_NUM_LAYERS", 2)
    config.setdefault("PQN_NORM_TYPE", "layer_norm")
    config.setdefault("PQN_DUELING", False)
    config.setdefault("PQN_APPEND_AGENT_ID", True)
    config.setdefault("PQN_ACTIVATION", "relu")
    config.setdefault("PQN_LAMBDA", 0.3)
    config.setdefault("PQN_EPS_START", 1.0)
    config.setdefault("PQN_EPS_FINISH", 0.1)
    config.setdefault("PQN_EPS_DECAY", 0.1)
    config.setdefault("PQN_ANNEAL_LR", False)
    config.setdefault("PQN_TEAM_REWARD", "sum")
    config.setdefault("PQN_NUM_MINIBATCHES", config.get("NUM_MINIBATCHES", 1))
    config.setdefault("PQN_UPDATE_EPOCHS", config.get("UPDATE_EPOCHS", 1))
    config["NUM_UPDATES"] = _compute_num_updates(
        config,
        algorithm_name=algorithm_name,
    )
    _validate_pqn_config(config, algorithm_name=algorithm_name)

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

    num_agents = len(env_spec.agent_ids)
    append_agent_id = bool(config["PQN_APPEND_AGENT_ID"])
    input_dim = env_spec.obs_dim + (num_agents if append_agent_id else 0)
    network = PQNQNetwork(
        action_dim=env_spec.action_dim,
        hidden_size=int(config["PQN_HIDDEN_SIZE"]),
        num_layers=int(config["PQN_NUM_LAYERS"]),
        norm_type=str(config["PQN_NORM_TYPE"]),
        activation=str(config["PQN_ACTIVATION"]),
        dueling=bool(config["PQN_DUELING"]),
    )

    total_grad_steps = max(
        int(config["NUM_UPDATES"])
        * int(config["PQN_UPDATE_EPOCHS"])
        * int(config["PQN_NUM_MINIBATCHES"]),
        1,
    )

    def linear_schedule(step_count: jax.Array) -> jax.Array:
        frac = 1.0 - step_count / total_grad_steps
        return jnp.asarray(config["LR"], dtype=jnp.float32) * frac

    if config.get("PQN_ANNEAL_LR", False):
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
        init_value=float(config["PQN_EPS_START"]),
        end_value=float(config["PQN_EPS_FINISH"]),
        transition_steps=max(
            int(float(config["PQN_EPS_DECAY"]) * max(int(config["NUM_UPDATES"]), 1)),
            1,
        ),
    )

    def apply_q(params: Any, obs: jax.Array) -> jax.Array:
        if append_agent_id:
            obs = _append_agent_ids(obs, num_agents=num_agents)
        leading_shape = obs.shape[:-1]
        flat_obs = jnp.reshape(obs, (-1, obs.shape[-1]))
        flat_q = network.apply(params, flat_obs)
        return jnp.reshape(flat_q, leading_shape + (env_spec.action_dim,))

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
    compute_next_team_q_jit = jax.jit(
        lambda params, next_obs, next_action_mask: jnp.sum(
            masked_max(apply_q(params, next_obs), next_action_mask),
            axis=-1,
        )
    )
    compute_targets_jit = jax.jit(
        lambda reward, terminated, next_team_q: _compute_q_lambda_targets(
            reward,
            terminated,
            next_team_q,
            gamma=jnp.asarray(config["GAMMA"], dtype=jnp.float32),
            trace_lambda=jnp.asarray(config["PQN_LAMBDA"], dtype=jnp.float32),
        )
    )
    update_minibatch_jit = jax.jit(
        lambda state, batch: _update_minibatch(
            network,
            tx,
            apply_q,
            state,
            batch,
        )
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        rng = jax.random.PRNGKey(int(rng)) if np.isscalar(rng) else rng
        num_envs = int(config["NUM_ENVS"])
        num_steps = int(config["NUM_STEPS"])
        batch_size = num_steps * num_envs
        minibatch_count = int(config["PQN_NUM_MINIBATCHES"])
        minibatch_size = batch_size // minibatch_count

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
                input_dim=input_dim,
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
                traj_team_rewards = np.zeros(
                    (num_steps, num_envs),
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
                    (num_steps, num_envs),
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
                    team_reward_batch = np.zeros(num_envs, dtype=np.float32)
                    terminated_batch = np.zeros(num_envs, dtype=np.float32)

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
                            phase="PQN-VDN rollout sampling",
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
                                "PQN-VDN expects all agents in an environment to "
                                "finish together."
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
                        if str(config["PQN_TEAM_REWARD"]).lower() == "sum":
                            team_reward_batch[env_idx] = float(reward_array.sum())
                        else:
                            team_reward_batch[env_idx] = float(reward_array.mean())
                        terminated_batch[env_idx] = float(terminated_array.max())
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
                    traj_team_rewards[step_idx] = team_reward_batch
                    traj_terminated[step_idx] = terminated_batch
                    last_obs = next_obs

                next_team_q = compute_next_team_q_jit(
                    train_state.params,
                    jnp.asarray(traj_next_obs),
                    jnp.asarray(traj_next_action_masks),
                )
                q_lambda_targets = compute_targets_jit(
                    jnp.asarray(traj_team_rewards),
                    jnp.asarray(traj_terminated),
                    next_team_q,
                )

                flat_obs = jnp.asarray(traj_obs).reshape(
                    (batch_size, num_agents, env_spec.obs_dim)
                )
                flat_actions = jnp.asarray(traj_actions).reshape(
                    (batch_size, num_agents)
                )
                flat_targets = jnp.asarray(q_lambda_targets).reshape((batch_size,))

                minibatch_metrics: list[dict[str, np.ndarray]] = []
                for _ in range(int(config["PQN_UPDATE_EPOCHS"])):
                    rng, permutation_rng = jax.random.split(rng)
                    permutation = jax.random.permutation(permutation_rng, batch_size)
                    shuffled_obs = flat_obs[permutation]
                    shuffled_actions = flat_actions[permutation]
                    shuffled_targets = flat_targets[permutation]
                    obs_minibatches = shuffled_obs.reshape(
                        (
                            minibatch_count,
                            minibatch_size,
                            num_agents,
                            env_spec.obs_dim,
                        )
                    )
                    action_minibatches = shuffled_actions.reshape(
                        (minibatch_count, minibatch_size, num_agents)
                    )
                    target_minibatches = shuffled_targets.reshape(
                        (minibatch_count, minibatch_size)
                    )
                    for minibatch_idx in range(minibatch_count):
                        train_state, loss_info = update_minibatch_jit(
                            train_state,
                            PQNVDNMinibatch(
                                obs=obs_minibatches[minibatch_idx],
                                action=action_minibatches[minibatch_idx],
                                target=target_minibatches[minibatch_idx],
                            ),
                        )
                        minibatch_metrics.append(
                            {
                                name: np.asarray(metric)
                                for name, metric in loss_info.items()
                            }
                        )

                aggregated_loss_metrics = {
                    key: np.stack(
                        [metric[key] for metric in minibatch_metrics],
                        axis=0,
                    ).mean(axis=0)
                    for key in minibatch_metrics[0]
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

                agent_q_value_mean = np.asarray(
                    aggregated_loss_metrics["agent_q_value_mean"],
                    dtype=np.float32,
                )
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
                    "q_loss": float(aggregated_loss_metrics["q_loss"]),
                    "td_error_mean": float(aggregated_loss_metrics["td_error_mean"]),
                    "q_value_mean": float(agent_q_value_mean.mean()),
                    "team_q_value_mean": float(
                        aggregated_loss_metrics["team_q_value_mean"]
                    ),
                    "target_mean": float(aggregated_loss_metrics["target_mean"]),
                }
                for agent_idx, agent_id in enumerate(env_spec.agent_ids):
                    metrics[f"{agent_id}/reward_mean"] = float(step_reward_mean[agent_idx])
                    metrics[f"{agent_id}/episode_return"] = float(
                        mean_episode_return[agent_idx]
                    )
                    metrics[f"{agent_id}/epsilon"] = epsilon
                    metrics[f"{agent_id}/q_value_mean"] = float(
                        agent_q_value_mean[agent_idx]
                    )

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
    q_values = apply_q(params, obs)
    greedy_actions = masked_argmax(q_values, action_mask)

    rng_random, rng_explore = jax.random.split(rng)
    random_actions = sample_masked_random_actions(action_mask, rng_random)
    explore = jax.random.uniform(rng_explore, greedy_actions.shape) < epsilon
    actions = jnp.where(explore, random_actions, greedy_actions)
    chosen_q_values = jnp.take_along_axis(
        q_values,
        actions[..., jnp.newaxis],
        axis=-1,
    ).squeeze(axis=-1)
    return actions, chosen_q_values


def _update_minibatch(
    network: PQNQNetwork,
    tx: optax.GradientTransformation,
    apply_q: Callable[[Any, jax.Array], jax.Array],
    train_state: PQNVDNTrainState,
    minibatch: PQNVDNMinibatch,
) -> tuple[PQNVDNTrainState, dict[str, jax.Array]]:
    del network

    def _loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        q_values = apply_q(params, minibatch.obs)
        chosen_q = jnp.take_along_axis(
            q_values,
            minibatch.action[..., jnp.newaxis],
            axis=-1,
        ).squeeze(axis=-1)
        team_q = jnp.sum(chosen_q, axis=-1)
        td_error = team_q - jax.lax.stop_gradient(minibatch.target)
        q_loss = jnp.mean(jnp.square(td_error))
        return q_loss, {
            "q_loss": q_loss,
            "td_error_mean": jnp.mean(jnp.abs(td_error)),
            "q_value_mean": jnp.mean(chosen_q),
            "team_q_value_mean": jnp.mean(team_q),
            "target_mean": jnp.mean(minibatch.target),
            "agent_q_value_mean": jnp.mean(chosen_q, axis=0),
        }

    (loss, loss_info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    updates, next_opt_state = tx.update(
        grads,
        train_state.opt_state,
        train_state.params,
    )
    next_params = optax.apply_updates(train_state.params, updates)

    metrics = {
        key: value
        for key, value in loss_info.items()
    }
    metrics["q_loss"] = loss

    return (
        PQNVDNTrainState(
            params=next_params,
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
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "MAX_GRAD_NORM": 0.5,
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

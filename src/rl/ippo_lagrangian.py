from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal

from .trajectory import (
    VectorEpisodicCostTracker,
    clipped_value_loss,
    compute_gae,
    discounted_cost_returns_at_starts,
    episode_history_entry,
    ppo_approx_kl,
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


class ActorCriticLagrangian(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
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

        reward_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        reward_hidden = activation(reward_hidden)
        reward_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(reward_hidden)
        reward_hidden = activation(reward_hidden)
        reward_critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(reward_hidden)

        cost_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        cost_hidden = activation(cost_hidden)
        cost_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(cost_hidden)
        cost_hidden = activation(cost_hidden)
        cost_critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(cost_hidden)

        return (
            actor_logits,
            jnp.squeeze(reward_critic, axis=-1),
            jnp.squeeze(cost_critic, axis=-1),
        )


class Transition(NamedTuple):
    terminated: jax.Array
    episode_done: jax.Array
    action: jax.Array
    reward_value: jax.Array
    cost_value: jax.Array
    next_reward_value: jax.Array
    next_cost_value: jax.Array
    reward: jax.Array
    cost: jax.Array
    log_prob: jax.Array
    obs: jax.Array


class MultiAgentLagrangianTrainState(NamedTuple):
    params: Any
    opt_state: Any
    lagrangian_multipliers: jax.Array


def init_train_state(
    rng: jax.Array,
    network: ActorCriticLagrangian,
    tx: optax.GradientTransformation,
    *,
    num_agents: int,
    obs_dim: int,
    lambda_init: float,
) -> MultiAgentLagrangianTrainState:
    init_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    init_rngs = jax.random.split(rng, num_agents)
    params = jax.vmap(network.init)(init_rngs, init_obs)
    opt_state = jax.vmap(tx.init)(params)
    lagrangian_multipliers = jnp.full(
        (num_agents,),
        jnp.asarray(lambda_init, dtype=jnp.float32),
    )
    return MultiAgentLagrangianTrainState(
        params=params,
        opt_state=opt_state,
        lagrangian_multipliers=lagrangian_multipliers,
    )


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    config = dict(config)
    config.setdefault("COST_LIMIT", 0.0)
    config.setdefault("LAMBDA_LR", 0.05)
    config.setdefault("LAMBDA_INIT", 0.0)
    config.setdefault("LAMBDA_MIN", 0.0)
    config.setdefault("LAMBDA_MAX", 1e6)
    config.setdefault("COST_GAMMA", config["GAMMA"])
    config.setdefault("COST_GAE_LAMBDA", config["GAE_LAMBDA"])
    config.setdefault("COST_BUDGET_GAMMA", 1.0)
    config.setdefault("COST_VF_COEF", config["VF_COEF"])
    config.setdefault("STANDARDIZE_REWARD_ADVANTAGES", True)
    config.setdefault("CENTER_COST_ADVANTAGES", True)
    config.setdefault("NORMALIZE_COST_ADVANTAGES", False)
    config.setdefault("COST_INFO_KEY", "safety_violation")
    config.setdefault("CLIP_VALUE_LOSS", True)
    config.setdefault("VALUE_CLIP_EPS", config["CLIP_EPS"])
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = (
        int(config["NUM_STEPS"])
        * int(config["NUM_ENVS"])
        // int(config["NUM_MINIBATCHES"])
    )
    if float(config["LAMBDA_MIN"]) < 0.0:
        raise ValueError("LAMBDA_MIN must be nonnegative.")
    if float(config["LAMBDA_MAX"]) < float(config["LAMBDA_MIN"]):
        raise ValueError("LAMBDA_MAX must be greater than or equal to LAMBDA_MIN.")
    if float(config["LAMBDA_LR"]) < 0.0:
        raise ValueError("LAMBDA_LR must be nonnegative.")
    config["LAMBDA_INIT"] = float(
        np.clip(
            float(config["LAMBDA_INIT"]),
            float(config["LAMBDA_MIN"]),
            float(config["LAMBDA_MAX"]),
        )
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

    network = ActorCriticLagrangian(
        env_spec.action_dim,
        activation=config.get("ACTIVATION", "tanh"),
    )

    def linear_schedule(step_count: jax.Array) -> jax.Array:
        minibatches_per_update = int(config["NUM_MINIBATCHES"]) * int(
            config["UPDATE_EPOCHS"]
        )
        frac = 1.0 - (step_count // minibatches_per_update) / max(
            config["NUM_UPDATES"], 1
        )
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

    def apply_model(
        params: Any,
        obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(lambda p, o: network.apply(p, o), in_axes=(0, 0))(params, obs)

    sample_action_jit = jax.jit(
        lambda params, obs, rng: _sample_actions(apply_model, params, obs, rng)
    )
    last_value_jit = jax.jit(
        lambda params, obs: _compute_last_values(apply_model, params, obs)
    )
    compute_advantages_jit = jax.jit(
        lambda traj_batch: _compute_advantages(
            config,
            traj_batch,
        )
    )
    update_minibatch_jit = jax.jit(
        lambda state, minibatch: _update_minibatch(
            config, network, tx, state, minibatch
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
        cost_tracker = VectorEpisodicCostTracker(
            num_envs=num_envs,
            num_agents=num_agents,
            gamma=float(config["COST_BUDGET_GAMMA"]),
            initial_returns=float(config["COST_LIMIT"]),
        )

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
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_actions = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.int32,
                )
                traj_log_probs = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_reward_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_cost_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_next_reward_values = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_next_cost_values = np.zeros(
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
                    rng, sample_rng = jax.random.split(rng)
                    actions, log_probs, reward_values, cost_values = sample_action_jit(
                        train_state.params,
                        obs_batch,
                        sample_rng,
                    )
                    actions_np = np.asarray(actions)
                    log_probs_np = np.asarray(log_probs)
                    reward_values_np = np.asarray(reward_values)
                    cost_values_np = np.asarray(cost_values)

                    next_obs = np.zeros_like(last_obs)
                    next_obs_for_value = np.zeros_like(last_obs)
                    reward_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    cost_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    terminated_batch = np.zeros(
                        (num_envs, num_agents), dtype=np.float32
                    )
                    episode_done_batch = np.zeros(
                        (num_envs, num_agents),
                        dtype=np.float32,
                    )

                    for env_idx, env in enumerate(envs):
                        env_global_step = (
                            update_idx * batch_size + step_idx * num_envs + env_idx + 1
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
                                "IPPO-Lagrangian expects all agents in an environment to finish together."
                            )

                        reward_batch[env_idx] = reward_array
                        cost_batch[env_idx] = cost_array
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
                        latest_cumulative_safety_violations[env_idx] = (
                            shared_info_value(
                                infos,
                                env_spec.agent_ids,
                                "safety_violations_cumulative",
                                default=0.0,
                            )
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
                            obs_dict, _ = env.reset(
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

                    traj_obs[step_idx] = last_obs
                    traj_actions[step_idx] = actions_np
                    traj_log_probs[step_idx] = log_probs_np
                    traj_reward_values[step_idx] = reward_values_np
                    traj_cost_values[step_idx] = cost_values_np
                    next_reward_values, next_cost_values = last_value_jit(
                        train_state.params,
                        jnp.asarray(next_obs_for_value),
                    )
                    traj_next_reward_values[step_idx] = np.asarray(next_reward_values)
                    traj_next_cost_values[step_idx] = np.asarray(next_cost_values)
                    traj_rewards[step_idx] = reward_batch
                    traj_costs[step_idx] = cost_batch
                    traj_terminated[step_idx] = terminated_batch
                    traj_episode_dones[step_idx] = episode_done_batch
                    last_obs = next_obs

                traj_batch = Transition(
                    terminated=jnp.asarray(traj_terminated),
                    episode_done=jnp.asarray(traj_episode_dones),
                    action=jnp.asarray(traj_actions),
                    reward_value=jnp.asarray(traj_reward_values),
                    cost_value=jnp.asarray(traj_cost_values),
                    next_reward_value=jnp.asarray(traj_next_reward_values),
                    next_cost_value=jnp.asarray(traj_next_cost_values),
                    reward=jnp.asarray(traj_rewards),
                    cost=jnp.asarray(traj_costs),
                    log_prob=jnp.asarray(traj_log_probs),
                    obs=jnp.asarray(traj_obs),
                )

                (
                    reward_advantages,
                    reward_targets,
                    cost_advantages,
                    cost_targets,
                ) = compute_advantages_jit(traj_batch)

                flat_batch = {
                    "obs": jnp.transpose(traj_batch.obs, (2, 0, 1, 3)).reshape(
                        num_agents, batch_size, env_spec.obs_dim
                    ),
                    "action": jnp.transpose(traj_batch.action, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "log_prob": jnp.transpose(traj_batch.log_prob, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "reward_value": jnp.transpose(
                        traj_batch.reward_value,
                        (2, 0, 1),
                    ).reshape(num_agents, batch_size),
                    "cost_value": jnp.transpose(
                        traj_batch.cost_value, (2, 0, 1)
                    ).reshape(num_agents, batch_size),
                    "reward_advantage": jnp.transpose(
                        reward_advantages,
                        (2, 0, 1),
                    ).reshape(num_agents, batch_size),
                    "cost_advantage": jnp.transpose(cost_advantages, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "reward_target": jnp.transpose(reward_targets, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                    "cost_target": jnp.transpose(cost_targets, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
                }

                (
                    mean_agent_cost_returns,
                    _,
                    completed_cost_episodes,
                ) = cost_tracker.add(traj_costs, traj_episode_dones)
                discounted_fragment_cost_returns = (
                    _mean_discounted_cost_returns_at_starts(
                        traj_costs,
                        traj_episode_dones,
                        gamma=float(config["COST_GAMMA"]),
                    )
                )
                previous_lagrangian_multipliers = np.asarray(
                    train_state.lagrangian_multipliers
                )
                proposed_lagrangian_multipliers = np.clip(
                    previous_lagrangian_multipliers
                    + float(config["LAMBDA_LR"])
                    * (mean_agent_cost_returns - float(config["COST_LIMIT"])),
                    float(config["LAMBDA_MIN"]),
                    float(config["LAMBDA_MAX"]),
                )
                dual_update_mask = completed_cost_episodes > 0
                lagrangian_multipliers_for_update = np.where(
                    dual_update_mask,
                    proposed_lagrangian_multipliers,
                    previous_lagrangian_multipliers,
                ).astype(np.float32)
                train_state = MultiAgentLagrangianTrainState(
                    params=train_state.params,
                    opt_state=train_state.opt_state,
                    lagrangian_multipliers=jnp.asarray(
                        lagrangian_multipliers_for_update,
                        dtype=jnp.float32,
                    ),
                )
                flat_batch["actor_advantage"] = _prepare_lagrangian_advantages(
                    flat_batch["reward_advantage"],
                    flat_batch["cost_advantage"],
                    train_state.lagrangian_multipliers,
                    standardize_reward=bool(config["STANDARDIZE_REWARD_ADVANTAGES"]),
                    center_cost=bool(config["CENTER_COST_ADVANTAGES"]),
                    normalize_cost=bool(config["NORMALIZE_COST_ADVANTAGES"]),
                )

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

                mean_agent_cost_rates = traj_costs.mean(axis=(0, 1))

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
                shield_interventions_mean = float(
                    traj_shield_intervention_counts.mean()
                )
                shield_interventions_agent_fraction_mean = float(
                    traj_shield_intervention_fractions.mean()
                )
                cumulative_safety_violations = float(
                    latest_cumulative_safety_violations.sum()
                )
                if completed_returns:
                    mean_episode_return = np.stack(completed_returns, axis=0).mean(
                        axis=0
                    )
                    mean_episode_length = float(np.mean(completed_lengths))
                    mean_episode_safety_violations = float(
                        np.mean(completed_safety_violation_counts)
                    )
                else:
                    mean_episode_return = np.zeros(num_agents, dtype=np.float32)
                    mean_episode_length = 0.0
                    mean_episode_safety_violations = 0.0

                lagrangian_multipliers = np.asarray(train_state.lagrangian_multipliers)
                constraint_violations = mean_agent_cost_returns - float(
                    config["COST_LIMIT"]
                )
                metrics = {
                    "update": float(update_idx),
                    "global_step": float((update_idx + 1) * batch_size),
                    "completed_episodes": float(total_completed_episodes),
                    "episode_length": mean_episode_length,
                    "reward_mean": float(step_reward_mean.mean()),
                    "episode_return_mean": float(mean_episode_return.mean()),
                    "cost_mean": float(mean_agent_cost_returns.mean()),
                    "cost_return_mean": float(mean_agent_cost_returns.mean()),
                    "discounted_cost_return_mean": float(
                        discounted_fragment_cost_returns.mean()
                    ),
                    "cost_rate_mean": float(mean_agent_cost_rates.mean()),
                    "constraint_violation_mean": float(constraint_violations.mean()),
                    "lagrangian_multiplier_mean": float(lagrangian_multipliers.mean()),
                    "lagrangian_multiplier_used_mean": float(
                        lagrangian_multipliers_for_update.mean()
                    ),
                    "cost_budget_fresh_fraction": float(dual_update_mask.mean()),
                    "dual_update_applied_fraction": float(dual_update_mask.mean()),
                    "completed_cost_episodes": float(completed_cost_episodes.mean()),
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
                    metrics[f"{agent_id}/reward_mean"] = float(
                        step_reward_mean[agent_idx]
                    )
                    metrics[f"{agent_id}/episode_return"] = float(
                        mean_episode_return[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_mean"] = float(
                        mean_agent_cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_return_mean"] = float(
                        mean_agent_cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/discounted_cost_return"] = float(
                        discounted_fragment_cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_rate_mean"] = float(
                        mean_agent_cost_rates[agent_idx]
                    )
                    metrics[f"{agent_id}/constraint_violation"] = float(
                        constraint_violations[agent_idx]
                    )
                    metrics[f"{agent_id}/lagrangian_multiplier"] = float(
                        lagrangian_multipliers[agent_idx]
                    )
                    metrics[f"{agent_id}/lagrangian_multiplier_used"] = float(
                        lagrangian_multipliers_for_update[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_budget_fresh"] = float(
                        dual_update_mask[agent_idx]
                    )
                    metrics[f"{agent_id}/dual_update_applied"] = float(
                        dual_update_mask[agent_idx]
                    )
                    metrics[f"{agent_id}/completed_cost_episodes"] = float(
                        completed_cost_episodes[agent_idx]
                    )
                    for key, value in aggregated_loss_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f} "
                        f"cost={metrics['cost_mean']:.3f} "
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
                    key: jnp.asarray(values) for key, values in metrics_history.items()
                },
                "episode_history": episode_history,
            }
        finally:
            for env in envs:
                env.close()

    return train


def _sample_actions(
    apply_model: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    params: Any,
    obs: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    logits, reward_values, cost_values = apply_model(params, obs_by_agent)
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
        jnp.swapaxes(reward_values, 0, 1),
        jnp.swapaxes(cost_values, 0, 1),
    )


def _compute_last_values(
    apply_model: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    params: Any,
    obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    _, reward_values, cost_values = apply_model(params, obs_by_agent)
    return jnp.swapaxes(reward_values, 0, 1), jnp.swapaxes(cost_values, 0, 1)


def _mean_discounted_cost_returns_at_starts(
    costs: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    return discounted_cost_returns_at_starts(costs, dones, gamma=gamma)


def _compute_advantages(
    config: Mapping[str, Any],
    traj_batch: Transition,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    reward_advantages, reward_targets = compute_gae(
        reward=traj_batch.reward,
        value=traj_batch.reward_value,
        next_value=traj_batch.next_reward_value,
        terminated=traj_batch.terminated,
        episode_done=traj_batch.episode_done,
        gamma=jnp.asarray(config["GAMMA"], dtype=jnp.float32),
        gae_lambda=jnp.asarray(config["GAE_LAMBDA"], dtype=jnp.float32),
    )
    cost_advantages, cost_targets = compute_gae(
        reward=traj_batch.cost,
        value=traj_batch.cost_value,
        next_value=traj_batch.next_cost_value,
        terminated=traj_batch.terminated,
        episode_done=traj_batch.episode_done,
        gamma=jnp.asarray(config["COST_GAMMA"], dtype=jnp.float32),
        gae_lambda=jnp.asarray(config["COST_GAE_LAMBDA"], dtype=jnp.float32),
    )
    return reward_advantages, reward_targets, cost_advantages, cost_targets


def _prepare_lagrangian_advantages(
    reward_advantage: jax.Array,
    cost_advantage: jax.Array,
    lagrangian_multipliers: jax.Array,
    *,
    standardize_reward: bool = True,
    center_cost: bool = True,
    normalize_cost: bool = False,
) -> jax.Array:
    """Prepare the full-rollout primal-dual advantage for PPO minibatches."""

    reward_for_actor = reward_advantage
    if standardize_reward:
        reward_for_actor = (
            reward_for_actor - reward_for_actor.mean(axis=1, keepdims=True)
        ) / (reward_for_actor.std(axis=1, keepdims=True) + 1e-8)

    cost_for_actor = cost_advantage
    if normalize_cost:
        cost_for_actor = (
            cost_for_actor - cost_for_actor.mean(axis=1, keepdims=True)
        ) / (cost_for_actor.std(axis=1, keepdims=True) + 1e-8)
    elif center_cost:
        cost_for_actor = cost_for_actor - cost_for_actor.mean(
            axis=1,
            keepdims=True,
        )

    multipliers = lagrangian_multipliers[:, None]
    return (reward_for_actor - multipliers * cost_for_actor) / (1.0 + multipliers)


def _update_minibatch(
    config: Mapping[str, Any],
    network: ActorCriticLagrangian,
    tx: optax.GradientTransformation,
    train_state: MultiAgentLagrangianTrainState,
    minibatch: Mapping[str, jax.Array],
) -> tuple[MultiAgentLagrangianTrainState, dict[str, jax.Array]]:
    clip_eps = jnp.asarray(config["CLIP_EPS"], dtype=jnp.float32)
    value_clip_eps = jnp.asarray(config["VALUE_CLIP_EPS"], dtype=jnp.float32)
    ent_coef = jnp.asarray(config["ENT_COEF"], dtype=jnp.float32)
    vf_coef = jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    cost_vf_coef = jnp.asarray(config["COST_VF_COEF"], dtype=jnp.float32)
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", True))

    def _loss_fn(
        params: Any,
        obs: jax.Array,
        action: jax.Array,
        old_log_prob: jax.Array,
        old_reward_value: jax.Array,
        old_cost_value: jax.Array,
        actor_advantage: jax.Array,
        reward_target: jax.Array,
        cost_target: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits, reward_value, cost_value = network.apply(params, obs)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(action)

        reward_value_loss = clipped_value_loss(
            value=reward_value,
            old_value=old_reward_value,
            target=reward_target,
            clip_eps=value_clip_eps,
            clip_value_loss=clip_value_loss,
        )

        cost_value_loss = clipped_value_loss(
            value=cost_value,
            old_value=old_cost_value,
            target=cost_target,
            clip_eps=value_clip_eps,
            clip_value_loss=clip_value_loss,
        )

        log_ratio = log_prob - old_log_prob
        ratio = jnp.exp(log_ratio)
        unclipped = ratio * actor_advantage
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * actor_advantage
        actor_loss = -jnp.minimum(unclipped, clipped).mean()
        entropy = pi.entropy().mean()
        total_loss = (
            actor_loss
            + vf_coef * reward_value_loss
            + cost_vf_coef * cost_value_loss
            - ent_coef * entropy
        )

        approx_kl = ppo_approx_kl(log_ratio)
        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": reward_value_loss,
            "cost_critic_loss": cost_value_loss,
            "entropy": entropy,
            "ratio": ratio.mean(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    grad_fn = jax.vmap(
        jax.value_and_grad(_loss_fn, has_aux=True),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    (losses, loss_info), grads = grad_fn(
        train_state.params,
        minibatch["obs"],
        minibatch["action"],
        minibatch["log_prob"],
        minibatch["reward_value"],
        minibatch["cost_value"],
        minibatch["actor_advantage"],
        minibatch["reward_target"],
        minibatch["cost_target"],
    )

    updates, next_opt_state = jax.vmap(tx.update)(
        grads,
        train_state.opt_state,
        train_state.params,
    )
    next_params = jax.vmap(optax.apply_updates)(train_state.params, updates)

    metrics = {key: value for key, value in loss_info.items()}
    metrics["total_loss"] = losses

    return (
        MultiAgentLagrangianTrainState(
            params=next_params,
            opt_state=next_opt_state,
            lagrangian_multipliers=train_state.lagrangian_multipliers,
        ),
        metrics,
    )

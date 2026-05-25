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
from jax.flatten_util import ravel_pytree

from .trajectory import (
    clipped_value_loss,
    compute_gae,
    discounted_cost_returns_and_scales_at_starts,
    discounted_cost_returns_at_starts,
    episode_history_entry,
    normalize_advantage,
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


class Actor(nn.Module):
    action_dim: int
    activation: str = "tanh"
    hidden_sizes: tuple[int, ...] = (64, 32)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        hidden = x
        for hidden_size in self.hidden_sizes:
            hidden = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(hidden)
            hidden = activation(hidden)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)


class Critic(nn.Module):
    activation: str = "tanh"
    hidden_sizes: tuple[int, ...] = (64, 32)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh

        hidden = x
        for hidden_size in self.hidden_sizes:
            hidden = nn.Dense(
                hidden_size,
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


class FailurePredictor(nn.Module):
    action_dim: int
    hidden_size: int = 32
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
    ) -> jax.Array:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        features = jnp.concatenate([obs, action_one_hot, next_obs], axis=-1)
        hidden = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(features)
        hidden = activation(hidden)
        logits = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        return jax.nn.sigmoid(jnp.squeeze(logits, axis=-1))


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
    next_obs: jax.Array


class ICPOTrainState(NamedTuple):
    actor_params: Any
    reward_critic_params: Any
    cost_critic_params: Any
    reward_critic_opt_state: Any
    cost_critic_opt_state: Any
    failure_predictor_params: Any
    failure_predictor_opt_state: Any


def init_train_state(
    rng: jax.Array,
    actor: Actor,
    critic: Critic,
    critic_tx: optax.GradientTransformation,
    failure_predictor: FailurePredictor,
    failure_predictor_tx: optax.GradientTransformation,
    *,
    num_agents: int,
    obs_dim: int,
) -> ICPOTrainState:
    init_obs = jnp.zeros((num_agents, obs_dim), dtype=jnp.float32)
    init_actions = jnp.zeros((num_agents,), dtype=jnp.int32)
    actor_rng, reward_critic_rng, cost_critic_rng, failure_predictor_rng = (
        jax.random.split(rng, 4)
    )
    actor_params = jax.vmap(actor.init)(
        jax.random.split(actor_rng, num_agents),
        init_obs,
    )
    reward_critic_params = jax.vmap(critic.init)(
        jax.random.split(reward_critic_rng, num_agents),
        init_obs,
    )
    cost_critic_params = jax.vmap(critic.init)(
        jax.random.split(cost_critic_rng, num_agents),
        init_obs,
    )
    failure_predictor_params = jax.vmap(failure_predictor.init)(
        jax.random.split(failure_predictor_rng, num_agents),
        init_obs,
        init_actions,
        init_obs,
    )
    reward_critic_opt_state = jax.vmap(critic_tx.init)(reward_critic_params)
    cost_critic_opt_state = jax.vmap(critic_tx.init)(cost_critic_params)
    failure_predictor_opt_state = jax.vmap(failure_predictor_tx.init)(
        failure_predictor_params
    )
    return ICPOTrainState(
        actor_params=actor_params,
        reward_critic_params=reward_critic_params,
        cost_critic_params=cost_critic_params,
        reward_critic_opt_state=reward_critic_opt_state,
        cost_critic_opt_state=cost_critic_opt_state,
        failure_predictor_params=failure_predictor_params,
        failure_predictor_opt_state=failure_predictor_opt_state,
    )


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    config = dict(config)
    config.setdefault("COST_LIMIT", 0.0)
    config.setdefault("COST_GAMMA", config["GAMMA"])
    config.setdefault("COST_GAE_LAMBDA", 0.5)
    config.setdefault("COST_VF_COEF", config["VF_COEF"])
    config.setdefault("COST_INFO_KEY", "safety_violation")
    config.setdefault("COST_SHAPING_ENABLED", True)
    config.setdefault("COST_SHAPING_HORIZON", 20)
    config.setdefault("COST_SHAPING_STEPS", 25)
    config.setdefault("COST_SHAPING_COEF", 1.0)
    config.setdefault("COST_SHAPING_LR", config["LR"])
    config.setdefault("COST_SHAPING_HIDDEN_SIZE", 32)
    if config["COST_SHAPING_LR"] is None:
        config["COST_SHAPING_LR"] = config["LR"]
    config.setdefault("CLIP_VALUE_LOSS", True)
    config.setdefault("VALUE_CLIP_EPS", config["CLIP_EPS"])
    config.setdefault("TARGET_KL", 0.01)
    config.setdefault("CG_ITERS", 10)
    config.setdefault("CG_DAMPING", 0.1)
    config.setdefault("FVP_SAMPLE_FREQ", 1)
    config.setdefault("LINE_SEARCH_STEPS", 15)
    config.setdefault("LINE_SEARCH_BACKTRACK_COEFF", 0.8)
    config.setdefault("ICPO_HIDDEN_SIZES", (64, 32))
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

    hidden_sizes = tuple(int(size) for size in config["ICPO_HIDDEN_SIZES"])
    actor = Actor(
        env_spec.action_dim,
        activation=config.get("ACTIVATION", "tanh"),
        hidden_sizes=hidden_sizes,
    )
    critic = Critic(
        activation=config.get("ACTIVATION", "tanh"),
        hidden_sizes=hidden_sizes,
    )
    failure_predictor = FailurePredictor(
        env_spec.action_dim,
        hidden_size=int(config["COST_SHAPING_HIDDEN_SIZE"]),
        activation=config.get("ACTIVATION", "tanh"),
    )

    def linear_schedule(step_count: jax.Array) -> jax.Array:
        minibatches_per_update = int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])
        frac = 1.0 - (step_count // minibatches_per_update) / max(config["NUM_UPDATES"], 1)
        return jnp.asarray(config["LR"], dtype=jnp.float32) * frac

    if config.get("ANNEAL_LR", False):
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    failure_predictor_tx = optax.adam(
        learning_rate=float(config["COST_SHAPING_LR"]),
        eps=1e-5,
    )

    def apply_actor(params: Any, obs: jax.Array) -> jax.Array:
        return jax.vmap(lambda p, o: actor.apply(p, o), in_axes=(0, 0))(params, obs)

    def apply_critics(
        reward_critic_params: Any,
        cost_critic_params: Any,
        obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        reward_values = jax.vmap(
            lambda p, o: critic.apply(p, o),
            in_axes=(0, 0),
        )(reward_critic_params, obs)
        cost_values = jax.vmap(
            lambda p, o: critic.apply(p, o),
            in_axes=(0, 0),
        )(cost_critic_params, obs)
        return reward_values, cost_values

    sample_action_jit = jax.jit(
        lambda actor_params, reward_critic_params, cost_critic_params, obs, rng: (
            _sample_actions(
                apply_actor,
                apply_critics,
                actor_params,
                reward_critic_params,
                cost_critic_params,
                obs,
                rng,
            )
        )
    )
    last_value_jit = jax.jit(
        lambda reward_critic_params, cost_critic_params, obs: _compute_last_values(
            apply_critics,
            reward_critic_params,
            cost_critic_params,
            obs,
        )
    )
    compute_advantages_jit = jax.jit(
        lambda traj_batch: _compute_advantages(
            config,
            traj_batch,
        )
    )
    update_failure_predictors_jit = jax.jit(
        lambda params, opt_state, predictor_batch: _update_failure_predictors(
            config,
            failure_predictor,
            failure_predictor_tx,
            params,
            opt_state,
            predictor_batch,
        )
    )
    predict_failure_deltas_jit = jax.jit(
        lambda params, predictor_batch: _predict_failure_deltas(
            failure_predictor,
            params,
            predictor_batch,
        )
    )
    update_actor_agent_jit = jax.jit(
        lambda params, agent_batch, cost_return, cost_return_scale: _update_actor_agent(
            config,
            actor,
            params,
            agent_batch,
            cost_return,
            cost_return_scale,
        )
    )
    update_critic_minibatch_jit = jax.jit(
        lambda state, minibatch: _update_critic_minibatch(
            config,
            critic,
            critic_tx,
            state,
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
                critic_tx,
                failure_predictor,
                failure_predictor_tx,
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
                traj_raw_costs = np.zeros(
                    (num_steps, num_envs, num_agents),
                    dtype=np.float32,
                )
                traj_next_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
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
                        train_state.actor_params,
                        train_state.reward_critic_params,
                        train_state.cost_critic_params,
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
                    raw_cost_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
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
                        raw_cost_array = dict_infos_to_array(
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
                                "ICPO expects all agents in an environment to finish together."
                            )

                        reward_batch[env_idx] = reward_array
                        raw_cost_batch[env_idx] = raw_cost_array
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
                        train_state.reward_critic_params,
                        train_state.cost_critic_params,
                        jnp.asarray(next_obs_for_value),
                    )
                    traj_next_reward_values[step_idx] = np.asarray(next_reward_values)
                    traj_next_cost_values[step_idx] = np.asarray(next_cost_values)
                    traj_rewards[step_idx] = reward_batch
                    traj_raw_costs[step_idx] = raw_cost_batch
                    traj_next_obs[step_idx] = next_obs_for_value
                    traj_terminated[step_idx] = terminated_batch
                    traj_episode_dones[step_idx] = episode_done_batch
                    last_obs = next_obs

                predictor_batch = {
                    "obs": jnp.transpose(jnp.asarray(traj_obs), (2, 0, 1, 3)).reshape(
                        num_agents,
                        batch_size,
                        env_spec.obs_dim,
                    ),
                    "action": jnp.transpose(
                        jnp.asarray(traj_actions),
                        (2, 0, 1),
                    ).reshape(num_agents, batch_size),
                    "next_obs": jnp.transpose(
                        jnp.asarray(traj_next_obs),
                        (2, 0, 1, 3),
                    ).reshape(num_agents, batch_size, env_spec.obs_dim),
                    "label": jnp.transpose(
                        jnp.asarray(
                            _future_violation_labels(
                                traj_raw_costs,
                                traj_episode_dones,
                                horizon=int(config["COST_SHAPING_HORIZON"]),
                            )
                        ),
                        (2, 0, 1),
                    ).reshape(num_agents, batch_size),
                }
                if bool(config["COST_SHAPING_ENABLED"]):
                    (
                        next_failure_predictor_params,
                        next_failure_predictor_opt_state,
                        shaping_info,
                    ) = update_failure_predictors_jit(
                        train_state.failure_predictor_params,
                        train_state.failure_predictor_opt_state,
                        predictor_batch,
                    )
                    train_state = ICPOTrainState(
                        actor_params=train_state.actor_params,
                        reward_critic_params=train_state.reward_critic_params,
                        cost_critic_params=train_state.cost_critic_params,
                        reward_critic_opt_state=train_state.reward_critic_opt_state,
                        cost_critic_opt_state=train_state.cost_critic_opt_state,
                        failure_predictor_params=next_failure_predictor_params,
                        failure_predictor_opt_state=next_failure_predictor_opt_state,
                    )
                    delta_by_agent = np.asarray(
                        predict_failure_deltas_jit(
                            train_state.failure_predictor_params,
                            predictor_batch,
                        )
                    )
                    shaping_loss_by_agent = np.asarray(shaping_info["cost_shaping_loss"])
                else:
                    delta_by_agent = np.zeros(
                        (num_agents, batch_size),
                        dtype=np.float32,
                    )
                    shaping_loss_by_agent = np.zeros(num_agents, dtype=np.float32)

                cost_deltas = np.transpose(
                    delta_by_agent.reshape(num_agents, num_steps, num_envs),
                    (1, 2, 0),
                ).astype(np.float32)
                cost_shaping_labels = np.transpose(
                    np.asarray(predictor_batch["label"]).reshape(
                        num_agents,
                        num_steps,
                        num_envs,
                    ),
                    (1, 2, 0),
                ).astype(np.float32)
                traj_costs = (
                    traj_raw_costs
                    + float(config["COST_SHAPING_COEF"]) * cost_deltas
                ).astype(np.float32)

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
                    next_obs=jnp.asarray(traj_next_obs),
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
                    "cost_value": jnp.transpose(traj_batch.cost_value, (2, 0, 1)).reshape(
                        num_agents, batch_size
                    ),
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

                cost_returns = _discounted_agent_cost_returns(
                    traj_costs,
                    traj_episode_dones,
                    gamma=float(config["COST_GAMMA"]),
                )
                raw_cost_returns = _discounted_agent_cost_returns(
                    traj_raw_costs,
                    traj_episode_dones,
                    gamma=float(config["COST_GAMMA"]),
                )
                cost_return_scales = _discounted_agent_cost_return_scales(
                    traj_costs,
                    traj_episode_dones,
                    gamma=float(config["COST_GAMMA"]),
                )
                next_actor_params, actor_metrics = _update_actors(
                    train_state.actor_params,
                    flat_batch,
                    jnp.asarray(cost_returns, dtype=jnp.float32),
                    jnp.asarray(cost_return_scales, dtype=jnp.float32),
                    update_actor_agent_jit,
                )
                train_state = ICPOTrainState(
                    actor_params=next_actor_params,
                    reward_critic_params=train_state.reward_critic_params,
                    cost_critic_params=train_state.cost_critic_params,
                    reward_critic_opt_state=train_state.reward_critic_opt_state,
                    cost_critic_opt_state=train_state.cost_critic_opt_state,
                    failure_predictor_params=train_state.failure_predictor_params,
                    failure_predictor_opt_state=train_state.failure_predictor_opt_state,
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
                        train_state, loss_info = update_critic_minibatch_jit(
                            train_state,
                            minibatch,
                        )
                        minibatch_metrics.append(
                            {
                                name: np.asarray(metric)
                                for name, metric in loss_info.items()
                            }
                        )

                aggregated_critic_metrics: dict[str, np.ndarray] = {}
                for key in minibatch_metrics[0]:
                    aggregated_critic_metrics[key] = np.stack(
                        [metric[key] for metric in minibatch_metrics],
                        axis=0,
                    ).mean(axis=0)

                step_reward_mean = traj_rewards.mean(axis=(0, 1))
                mean_agent_costs = traj_costs.mean(axis=(0, 1))
                mean_agent_raw_costs = traj_raw_costs.mean(axis=(0, 1))
                mean_agent_cost_deltas = cost_deltas.mean(axis=(0, 1))
                mean_agent_cost_labels = cost_shaping_labels.mean(axis=(0, 1))
                constraint_violations = cost_returns - float(config["COST_LIMIT"])
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

                actor_metrics_np = {
                    key: np.asarray(value)
                    for key, value in actor_metrics.items()
                }
                metrics = {
                    "update": float(update_idx),
                    "global_step": float((update_idx + 1) * batch_size),
                    "completed_episodes": float(total_completed_episodes),
                    "episode_length": mean_episode_length,
                    "reward_mean": float(step_reward_mean.mean()),
                    "episode_return_mean": float(mean_episode_return.mean()),
                    "cost_mean": float(mean_agent_costs.mean()),
                    "cost_return_mean": float(cost_returns.mean()),
                    "raw_cost_mean": float(mean_agent_raw_costs.mean()),
                    "raw_cost_return_mean": float(raw_cost_returns.mean()),
                    "shaped_cost_mean": float(mean_agent_costs.mean()),
                    "shaped_cost_return_mean": float(cost_returns.mean()),
                    "cost_shaping_delta_mean": float(mean_agent_cost_deltas.mean()),
                    "cost_shaping_label_mean": float(mean_agent_cost_labels.mean()),
                    "cost_shaping_loss": float(shaping_loss_by_agent.mean()),
                    "constraint_violation_mean": float(constraint_violations.mean()),
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
                    "cpo_mean_kl": float(actor_metrics_np["cpo_mean_kl"].mean()),
                    "cpo_accepted_fraction": float(
                        actor_metrics_np["cpo_accepted"].mean()
                    ),
                    "cpo_step_fraction_mean": float(
                        actor_metrics_np["cpo_step_fraction"].mean()
                    ),
                    "cpo_constraint_scale_mean": float(
                        actor_metrics_np["cpo_constraint_scale"].mean()
                    ),
                    "cpo_scaled_constraint_violation_mean": float(
                        actor_metrics_np["cpo_scaled_constraint_violation"].mean()
                    ),
                    "cpo_optim_case_mean": float(
                        actor_metrics_np["cpo_optim_case"].mean()
                    ),
                    "cpo_lambda_mean": float(actor_metrics_np["cpo_lambda"].mean()),
                    "cpo_nu_mean": float(actor_metrics_np["cpo_nu"].mean()),
                }

                for key, value in aggregated_critic_metrics.items():
                    metrics[key] = float(value.mean())
                for key in (
                    "cpo_step_norm",
                    "cpo_reward_gradient_norm",
                    "cpo_cost_gradient_norm",
                    "cpo_reward_cg_residual_norm",
                    "cpo_cost_cg_residual_norm",
                    "cpo_line_search_rejections",
                    "cpo_line_search_attempts",
                    "cpo_final_surrogate_constraint",
                ):
                    metrics[key] = float(actor_metrics_np[key].mean())
                for agent_idx, agent_id in enumerate(env_spec.agent_ids):
                    metrics[f"{agent_id}/reward_mean"] = float(step_reward_mean[agent_idx])
                    metrics[f"{agent_id}/episode_return"] = float(
                        mean_episode_return[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_mean"] = float(mean_agent_costs[agent_idx])
                    metrics[f"{agent_id}/raw_cost_mean"] = float(
                        mean_agent_raw_costs[agent_idx]
                    )
                    metrics[f"{agent_id}/raw_cost_return"] = float(
                        raw_cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/raw_cost_return_mean"] = float(
                        raw_cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/shaped_cost_mean"] = float(
                        mean_agent_costs[agent_idx]
                    )
                    metrics[f"{agent_id}/shaped_cost_return"] = float(
                        cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/shaped_cost_return_mean"] = float(
                        cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_shaping_delta_mean"] = float(
                        mean_agent_cost_deltas[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_shaping_label_mean"] = float(
                        mean_agent_cost_labels[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_shaping_loss"] = float(
                        shaping_loss_by_agent[agent_idx]
                    )
                    metrics[f"{agent_id}/cost_return"] = float(cost_returns[agent_idx])
                    metrics[f"{agent_id}/cost_return_mean"] = float(
                        cost_returns[agent_idx]
                    )
                    metrics[f"{agent_id}/constraint_violation"] = float(
                        constraint_violations[agent_idx]
                    )
                    for key, value in aggregated_critic_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])
                    for key, value in actor_metrics_np.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f} "
                        f"cost_return={metrics['cost_return_mean']:.3f} "
                        f"kl={metrics['cpo_mean_kl']:.5f}"
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
    apply_critics: Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]],
    actor_params: Any,
    reward_critic_params: Any,
    cost_critic_params: Any,
    obs: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    logits = apply_actor(actor_params, obs_by_agent)
    reward_values, cost_values = apply_critics(
        reward_critic_params,
        cost_critic_params,
        obs_by_agent,
    )
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
    apply_critics: Callable[[Any, Any, jax.Array], tuple[jax.Array, jax.Array]],
    reward_critic_params: Any,
    cost_critic_params: Any,
    obs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    obs_by_agent = jnp.swapaxes(obs, 0, 1)
    reward_values, cost_values = apply_critics(
        reward_critic_params,
        cost_critic_params,
        obs_by_agent,
    )
    return jnp.swapaxes(reward_values, 0, 1), jnp.swapaxes(cost_values, 0, 1)


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


def _discounted_agent_cost_returns(
    costs: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    return discounted_cost_returns_at_starts(costs, dones, gamma=gamma)


def _discounted_agent_cost_return_scales(
    costs: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    _, scales = discounted_cost_returns_and_scales_at_starts(
        costs,
        dones,
        gamma=gamma,
    )
    return scales


def _future_violation_labels(
    raw_costs: np.ndarray,
    episode_dones: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    raw_costs = np.asarray(raw_costs, dtype=np.float32)
    episode_dones = np.asarray(episode_dones, dtype=np.float32)
    if raw_costs.shape != episode_dones.shape:
        raise ValueError("raw_costs and episode_dones must have matching shapes.")

    labels = np.zeros_like(raw_costs, dtype=np.float32)
    if horizon <= 0:
        return labels

    num_steps, num_envs, num_agents = raw_costs.shape
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            for step_idx in range(num_steps):
                saw_violation = False
                for offset in range(horizon):
                    future_idx = step_idx + offset
                    if future_idx >= num_steps:
                        break
                    saw_violation = saw_violation or (
                        raw_costs[future_idx, env_idx, agent_idx] > 0.0
                    )
                    if episode_dones[future_idx, env_idx, agent_idx] > 0.0:
                        break
                labels[step_idx, env_idx, agent_idx] = float(saw_violation)
    return labels


def _predictor_binary_cross_entropy(
    probabilities: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    eps = jnp.asarray(1e-6, dtype=probabilities.dtype)
    clipped = jnp.clip(probabilities, eps, 1.0 - eps)
    return -jnp.mean(
        labels * jnp.log(clipped)
        + (1.0 - labels) * jnp.log(1.0 - clipped)
    )


def _failure_predictor_loss(
    failure_predictor: FailurePredictor,
    params: Any,
    obs: jax.Array,
    action: jax.Array,
    next_obs: jax.Array,
    label: jax.Array,
) -> jax.Array:
    probabilities = failure_predictor.apply(params, obs, action, next_obs)
    return _predictor_binary_cross_entropy(probabilities, label)


def _update_failure_predictors(
    config: Mapping[str, Any],
    failure_predictor: FailurePredictor,
    failure_predictor_tx: optax.GradientTransformation,
    params: Any,
    opt_state: Any,
    predictor_batch: Mapping[str, jax.Array],
) -> tuple[Any, Any, dict[str, jax.Array]]:
    shaping_steps = int(config["COST_SHAPING_STEPS"])

    def _loss_fn(
        one_params: Any,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        label: jax.Array,
    ) -> jax.Array:
        return _failure_predictor_loss(
            failure_predictor,
            one_params,
            obs,
            action,
            next_obs,
            label,
        )

    if shaping_steps <= 0:
        losses = jax.vmap(_loss_fn)(
            params,
            predictor_batch["obs"],
            predictor_batch["action"],
            predictor_batch["next_obs"],
            predictor_batch["label"],
        )
        return params, opt_state, {"cost_shaping_loss": losses.astype(jnp.float32)}

    def _update_one(
        one_params: Any,
        one_opt_state: Any,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        label: jax.Array,
    ) -> tuple[Any, Any, jax.Array]:
        def _step(carry: tuple[Any, Any], _: None) -> tuple[tuple[Any, Any], jax.Array]:
            step_params, step_opt_state = carry
            loss, grads = jax.value_and_grad(_loss_fn)(
                step_params,
                obs,
                action,
                next_obs,
                label,
            )
            updates, next_opt_state = failure_predictor_tx.update(
                grads,
                step_opt_state,
                step_params,
            )
            next_params = optax.apply_updates(step_params, updates)
            return (next_params, next_opt_state), loss

        (next_params, next_opt_state), losses = jax.lax.scan(
            _step,
            (one_params, one_opt_state),
            None,
            length=shaping_steps,
        )
        return next_params, next_opt_state, losses[-1]

    next_params, next_opt_state, losses = jax.vmap(_update_one)(
        params,
        opt_state,
        predictor_batch["obs"],
        predictor_batch["action"],
        predictor_batch["next_obs"],
        predictor_batch["label"],
    )
    return (
        next_params,
        next_opt_state,
        {"cost_shaping_loss": losses.astype(jnp.float32)},
    )


def _predict_failure_deltas(
    failure_predictor: FailurePredictor,
    params: Any,
    predictor_batch: Mapping[str, jax.Array],
) -> jax.Array:
    return jax.vmap(
        lambda one_params, obs, action, next_obs: failure_predictor.apply(
            one_params,
            obs,
            action,
            next_obs,
        )
    )(
        params,
        predictor_batch["obs"],
        predictor_batch["action"],
        predictor_batch["next_obs"],
    )


def _tree_take(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _tree_stack(trees: list[Any]) -> Any:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *trees)


def _update_actors(
    actor_params: Any,
    flat_batch: Mapping[str, jax.Array],
    cost_returns: jax.Array,
    cost_return_scales: jax.Array,
    update_actor_agent: Callable[
        [Any, Mapping[str, jax.Array], jax.Array, jax.Array],
        tuple[Any, dict[str, jax.Array]],
    ],
) -> tuple[Any, dict[str, jax.Array]]:
    num_agents = int(cost_returns.shape[0])
    next_params_by_agent = []
    metrics_by_agent = []
    for agent_idx in range(num_agents):
        agent_params = _tree_take(actor_params, agent_idx)
        agent_batch = {
            key: value[agent_idx]
            for key, value in flat_batch.items()
            if key in {"obs", "action", "log_prob", "reward_advantage", "cost_advantage"}
        }
        next_agent_params, agent_metrics = update_actor_agent(
            agent_params,
            agent_batch,
            cost_returns[agent_idx],
            cost_return_scales[agent_idx],
        )
        next_params_by_agent.append(next_agent_params)
        metrics_by_agent.append(agent_metrics)

    next_actor_params = _tree_stack(next_params_by_agent)
    metrics = {
        key: jnp.asarray([agent_metrics[key] for agent_metrics in metrics_by_agent])
        for key in metrics_by_agent[0]
    }
    return next_actor_params, metrics


def _normalized_advantage(advantage: jax.Array) -> jax.Array:
    return normalize_advantage(advantage)


def _actor_surrogates(
    actor: Actor,
    params: Any,
    obs: jax.Array,
    action: jax.Array,
    old_log_prob: jax.Array,
    reward_advantage: jax.Array,
    cost_advantage: jax.Array,
    old_logits: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    logits = actor.apply(params, obs)
    pi = distrax.Categorical(logits=logits)
    log_prob = pi.log_prob(action)
    ratio = jnp.exp(log_prob - old_log_prob)
    reward_objective = (ratio * _normalized_advantage(reward_advantage)).mean()
    cost_objective = (ratio * cost_advantage).mean()
    old_pi = distrax.Categorical(logits=old_logits)
    mean_kl = pi.kl_divergence(old_pi).mean()
    entropy = pi.entropy().mean()
    return reward_objective, cost_objective, mean_kl, entropy


def _conjugate_gradient(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    cg_iters: int,
    residual_tol: float = 1e-10,
) -> jax.Array:
    x = jnp.zeros_like(b)
    r = b.copy()
    p = b.copy()
    rdotr = jnp.dot(r, r)
    eps = jnp.asarray(1e-8, dtype=b.dtype)
    tol = jnp.asarray(residual_tol, dtype=b.dtype)

    def _body(_, carry):
        x, r, p, rdotr = carry
        z = matvec(p)
        alpha = rdotr / (jnp.dot(p, z) + eps)
        next_x = x + alpha * p
        next_r = r - alpha * z
        next_rdotr = jnp.dot(next_r, next_r)
        beta = next_rdotr / (rdotr + eps)
        next_p = next_r + beta * p
        should_update = rdotr > tol
        return (
            jnp.where(should_update, next_x, x),
            jnp.where(should_update, next_r, r),
            jnp.where(should_update, next_p, p),
            jnp.where(should_update, next_rdotr, rdotr),
        )

    x, _, _, _ = jax.lax.fori_loop(
        0,
        int(cg_iters),
        _body,
        (x, r, p, rdotr),
    )
    return x


def _compute_cpo_step_direction(
    *,
    x: jax.Array,
    p: jax.Array,
    q: jax.Array,
    r: jax.Array,
    s: jax.Array,
    constraint_violation: jax.Array,
    target_kl: jax.Array,
    cost_grad_norm_sq: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    eps = jnp.asarray(1e-8, dtype=x.dtype)
    delta = 2.0 * target_kl
    c = constraint_violation
    s_safe = s + eps
    q_safe = q + eps
    A = jnp.maximum(q - jnp.square(r) / s_safe, 0.0)
    B = delta - jnp.square(c) / s_safe
    no_cost_gradient = cost_grad_norm_sq <= 1e-12

    optim_case = jnp.where(
        no_cost_gradient & (c < 0.0),
        4,
        jnp.where(
            (c < 0.0) & (B < 0.0),
            3,
            jnp.where(
                (c < 0.0) & (B >= 0.0),
                2,
                jnp.where((c >= 0.0) & (B >= 0.0), 1, 0),
            ),
        ),
    )

    trpo_scale = jnp.sqrt(delta / q_safe)
    trpo_step = trpo_scale * x
    trpo_lambda = 1.0 / (trpo_scale + eps)

    B_nonnegative = jnp.maximum(B, 0.0)
    lambda_a = jnp.sqrt(A / (B_nonnegative + eps))
    lambda_b = jnp.sqrt(q_safe / (delta + eps))
    trpo_cost = c + trpo_scale * r
    trpo_is_feasible = trpo_cost <= 0.0

    tangent_step = x - (r / s_safe) * p
    tangent_scale = jnp.sqrt(B_nonnegative / (A + eps))
    active_step = -(c / s_safe) * p + tangent_scale * tangent_step
    active_lambda = jnp.maximum(lambda_a, eps)
    active_nu = jnp.maximum(0.0, r + active_lambda * c) / s_safe

    constrained_step = jnp.where(trpo_is_feasible, trpo_step, active_step)
    constrained_lambda = jnp.where(trpo_is_feasible, lambda_b, active_lambda)
    constrained_nu = jnp.where(trpo_is_feasible, 0.0, active_nu)

    recovery_nu = jnp.sqrt(delta / s_safe)
    recovery_step = -recovery_nu * p

    use_trpo = (optim_case == 3) | (optim_case == 4)
    use_recovery = optim_case == 0
    step_direction = jnp.where(
        use_trpo,
        trpo_step,
        jnp.where(use_recovery, recovery_step, constrained_step),
    )
    lambda_star = jnp.where(
        use_trpo,
        trpo_lambda,
        jnp.where(use_recovery, 0.0, constrained_lambda),
    )
    nu_star = jnp.where(
        use_trpo,
        0.0,
        jnp.where(use_recovery, recovery_nu, constrained_nu),
    )
    step_direction = jnp.where(
        jnp.isfinite(step_direction),
        step_direction,
        jnp.zeros_like(step_direction),
    )
    return step_direction, {
        "cpo_optim_case": optim_case.astype(jnp.float32),
        "cpo_lambda": lambda_star.astype(jnp.float32),
        "cpo_nu": nu_star.astype(jnp.float32),
        "cpo_q": q.astype(jnp.float32),
        "cpo_r": r.astype(jnp.float32),
        "cpo_s": s.astype(jnp.float32),
        "cpo_A": A.astype(jnp.float32),
        "cpo_B": B.astype(jnp.float32),
    }


def _line_search_actor(
    actor: Actor,
    params: Any,
    flat_step: jax.Array,
    agent_batch: Mapping[str, jax.Array],
    old_logits: jax.Array,
    constraint_violation: jax.Array,
    optim_case: jax.Array,
    *,
    target_kl: jax.Array,
    line_search_steps: int,
    backtrack_coeff: float,
) -> tuple[Any, dict[str, jax.Array]]:
    old_flat, unravel = ravel_pytree(params)
    old_reward_objective, old_cost_objective, _, _ = _actor_surrogates(
        actor,
        params,
        agent_batch["obs"],
        agent_batch["action"],
        agent_batch["log_prob"],
        agent_batch["reward_advantage"],
        agent_batch["cost_advantage"],
        old_logits,
    )

    initial_carry = (
        jnp.asarray(False),
        old_flat,
        jnp.asarray(0.0, dtype=old_flat.dtype),
        jnp.asarray(0.0, dtype=old_flat.dtype),
        jnp.asarray(0.0, dtype=old_flat.dtype),
        jnp.asarray(0.0, dtype=old_flat.dtype),
        jnp.asarray(0.0, dtype=old_flat.dtype),
    )

    def _body(carry, step_idx):
        (
            accepted,
            best_flat,
            best_fraction,
            best_kl,
            best_reward_improve,
            best_cost_diff,
            reject_count,
        ) = carry
        step_fraction = jnp.power(
            jnp.asarray(backtrack_coeff, dtype=old_flat.dtype),
            step_idx.astype(old_flat.dtype),
        )
        candidate_flat = old_flat + step_fraction * flat_step
        candidate_params = unravel(candidate_flat)
        reward_objective, cost_objective, mean_kl, _ = _actor_surrogates(
            actor,
            candidate_params,
            agent_batch["obs"],
            agent_batch["action"],
            agent_batch["log_prob"],
            agent_batch["reward_advantage"],
            agent_batch["cost_advantage"],
            old_logits,
        )
        reward_improve = reward_objective - old_reward_objective
        cost_diff = cost_objective - old_cost_objective
        reward_ok = jnp.logical_or(optim_case <= 1.0, reward_improve >= -1e-10)
        is_recovery = optim_case == 0.0
        cost_ok = jnp.where(
            is_recovery,
            cost_diff < -1e-10,
            constraint_violation + cost_diff <= 1e-10,
        )
        kl_ok = mean_kl <= target_kl
        finite = (
            jnp.isfinite(reward_objective)
            & jnp.isfinite(cost_objective)
            & jnp.isfinite(mean_kl)
        )
        should_accept = (~accepted) & reward_ok & cost_ok & kl_ok & finite
        should_reject = (~accepted) & (~should_accept)
        return (
            accepted | should_accept,
            jnp.where(should_accept, candidate_flat, best_flat),
            jnp.where(should_accept, step_fraction, best_fraction),
            jnp.where(should_accept, mean_kl, best_kl),
            jnp.where(should_accept, reward_improve, best_reward_improve),
            jnp.where(should_accept, cost_diff, best_cost_diff),
            reject_count + should_reject.astype(old_flat.dtype),
        ), None

    (
        accepted,
        best_flat,
        best_fraction,
        best_kl,
        best_reward_improve,
        best_cost_diff,
        reject_count,
    ), _ = jax.lax.scan(
        _body,
        initial_carry,
        jnp.arange(int(line_search_steps)),
    )
    accepted_float = accepted.astype(jnp.float32)
    final_surrogate_constraint = constraint_violation + best_cost_diff
    return unravel(best_flat), {
        "cpo_accepted": accepted_float,
        "cpo_step_fraction": best_fraction.astype(jnp.float32),
        "cpo_mean_kl": best_kl.astype(jnp.float32),
        "cpo_reward_improvement": best_reward_improve.astype(jnp.float32),
        "cpo_cost_diff": best_cost_diff.astype(jnp.float32),
        "cpo_line_search_rejections": reject_count.astype(jnp.float32),
        "cpo_line_search_attempts": (reject_count + accepted_float).astype(jnp.float32),
        "cpo_final_surrogate_constraint": final_surrogate_constraint.astype(
            jnp.float32
        ),
    }


def _update_actor_agent(
    config: Mapping[str, Any],
    actor: Actor,
    params: Any,
    agent_batch: Mapping[str, jax.Array],
    cost_return: jax.Array,
    cost_return_scale: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    target_kl = jnp.asarray(config["TARGET_KL"], dtype=jnp.float32)
    cg_damping = jnp.asarray(config["CG_DAMPING"], dtype=jnp.float32)
    cost_limit = jnp.asarray(config["COST_LIMIT"], dtype=jnp.float32)
    fvp_sample_freq = max(1, int(config.get("FVP_SAMPLE_FREQ", 1)))
    old_logits = actor.apply(params, agent_batch["obs"])
    fvp_obs = agent_batch["obs"][::fvp_sample_freq]
    fvp_old_logits = old_logits[::fvp_sample_freq]

    def _reward_objective(current_params: Any) -> jax.Array:
        return _actor_surrogates(
            actor,
            current_params,
            agent_batch["obs"],
            agent_batch["action"],
            agent_batch["log_prob"],
            agent_batch["reward_advantage"],
            agent_batch["cost_advantage"],
            old_logits,
        )[0]

    def _cost_objective(current_params: Any) -> jax.Array:
        return _actor_surrogates(
            actor,
            current_params,
            agent_batch["obs"],
            agent_batch["action"],
            agent_batch["log_prob"],
            agent_batch["reward_advantage"],
            agent_batch["cost_advantage"],
            old_logits,
        )[1]

    reward_grad = jax.grad(_reward_objective)(params)
    cost_grad = jax.grad(_cost_objective)(params)
    flat_params, unravel = ravel_pytree(params)
    flat_reward_grad, _ = ravel_pytree(reward_grad)
    flat_cost_grad, _ = ravel_pytree(cost_grad)

    def _kl_flat(flat_actor_params: jax.Array) -> jax.Array:
        current_params = unravel(flat_actor_params)
        logits = actor.apply(current_params, fvp_obs)
        old_pi = distrax.Categorical(logits=fvp_old_logits)
        current_pi = distrax.Categorical(logits=logits)
        return current_pi.kl_divergence(old_pi).mean()

    grad_kl_flat = jax.grad(_kl_flat)

    def _hvp(vector: jax.Array) -> jax.Array:
        hvp = jax.jvp(grad_kl_flat, (flat_params,), (vector,))[1]
        return hvp + cg_damping * vector

    x = _conjugate_gradient(_hvp, flat_reward_grad, int(config["CG_ITERS"]))
    p = _conjugate_gradient(_hvp, flat_cost_grad, int(config["CG_ITERS"]))
    q = jnp.maximum(jnp.dot(x, _hvp(x)), 0.0)
    r = jnp.dot(flat_reward_grad, p)
    s = jnp.maximum(jnp.dot(flat_cost_grad, p), 0.0)
    raw_constraint_violation = cost_return - cost_limit
    constraint_violation = raw_constraint_violation * cost_return_scale
    flat_step, cpo_info = _compute_cpo_step_direction(
        x=x,
        p=p,
        q=q,
        r=r,
        s=s,
        constraint_violation=constraint_violation,
        target_kl=target_kl,
        cost_grad_norm_sq=jnp.dot(flat_cost_grad, flat_cost_grad),
    )
    reward_cg_residual = jnp.linalg.norm(_hvp(x) - flat_reward_grad)
    cost_cg_residual = jnp.linalg.norm(_hvp(p) - flat_cost_grad)
    next_params, line_search_info = _line_search_actor(
        actor,
        params,
        flat_step,
        agent_batch,
        old_logits,
        constraint_violation,
        cpo_info["cpo_optim_case"],
        target_kl=target_kl,
        line_search_steps=int(config["LINE_SEARCH_STEPS"]),
        backtrack_coeff=float(config["LINE_SEARCH_BACKTRACK_COEFF"]),
    )
    _, _, _, entropy = _actor_surrogates(
        actor,
        params,
        agent_batch["obs"],
        agent_batch["action"],
        agent_batch["log_prob"],
        agent_batch["reward_advantage"],
        agent_batch["cost_advantage"],
        old_logits,
    )
    metrics = {
        **cpo_info,
        **line_search_info,
        "entropy": entropy.astype(jnp.float32),
        "actor_reward_surrogate": _reward_objective(params).astype(jnp.float32),
        "actor_cost_surrogate": _cost_objective(params).astype(jnp.float32),
        "cpo_constraint_scale": cost_return_scale.astype(jnp.float32),
        "cpo_scaled_constraint_violation": constraint_violation.astype(jnp.float32),
        "cpo_step_norm": jnp.linalg.norm(flat_step).astype(jnp.float32),
        "cpo_reward_gradient_norm": jnp.linalg.norm(flat_reward_grad).astype(jnp.float32),
        "cpo_cost_gradient_norm": jnp.linalg.norm(flat_cost_grad).astype(jnp.float32),
        "cpo_reward_cg_residual_norm": reward_cg_residual.astype(jnp.float32),
        "cpo_cost_cg_residual_norm": cost_cg_residual.astype(jnp.float32),
    }
    return next_params, metrics


def _update_critic_minibatch(
    config: Mapping[str, Any],
    critic: Critic,
    critic_tx: optax.GradientTransformation,
    train_state: ICPOTrainState,
    minibatch: Mapping[str, jax.Array],
) -> tuple[ICPOTrainState, dict[str, jax.Array]]:
    value_clip_eps = jnp.asarray(config["VALUE_CLIP_EPS"], dtype=jnp.float32)
    vf_coef = jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    cost_vf_coef = jnp.asarray(config["COST_VF_COEF"], dtype=jnp.float32)
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", True))

    def _critic_loss_fn(
        params: Any,
        obs: jax.Array,
        old_value: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        value = critic.apply(params, obs)
        return clipped_value_loss(
            value=value,
            old_value=old_value,
            target=target,
            clip_eps=value_clip_eps,
            clip_value_loss=clip_value_loss,
        )

    reward_grad_fn = jax.vmap(
        jax.value_and_grad(_critic_loss_fn),
        in_axes=(0, 0, 0, 0),
    )
    reward_losses, reward_grads = reward_grad_fn(
        train_state.reward_critic_params,
        minibatch["obs"],
        minibatch["reward_value"],
        minibatch["reward_target"],
    )
    reward_updates, next_reward_opt_state = jax.vmap(critic_tx.update)(
        jax.tree.map(lambda grad: vf_coef * grad, reward_grads),
        train_state.reward_critic_opt_state,
        train_state.reward_critic_params,
    )
    next_reward_params = jax.vmap(optax.apply_updates)(
        train_state.reward_critic_params,
        reward_updates,
    )

    cost_grad_fn = jax.vmap(
        jax.value_and_grad(_critic_loss_fn),
        in_axes=(0, 0, 0, 0),
    )
    cost_losses, cost_grads = cost_grad_fn(
        train_state.cost_critic_params,
        minibatch["obs"],
        minibatch["cost_value"],
        minibatch["cost_target"],
    )
    cost_updates, next_cost_opt_state = jax.vmap(critic_tx.update)(
        jax.tree.map(lambda grad: cost_vf_coef * grad, cost_grads),
        train_state.cost_critic_opt_state,
        train_state.cost_critic_params,
    )
    next_cost_params = jax.vmap(optax.apply_updates)(
        train_state.cost_critic_params,
        cost_updates,
    )

    return (
        ICPOTrainState(
            actor_params=train_state.actor_params,
            reward_critic_params=next_reward_params,
            cost_critic_params=next_cost_params,
            reward_critic_opt_state=next_reward_opt_state,
            cost_critic_opt_state=next_cost_opt_state,
            failure_predictor_params=train_state.failure_predictor_params,
            failure_predictor_opt_state=train_state.failure_predictor_opt_state,
        ),
        {
            "critic_loss": reward_losses,
            "cost_critic_loss": cost_losses,
        },
    )

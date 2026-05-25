from __future__ import annotations

from collections import defaultdict
from itertools import product
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
    dict_values_to_array,
    mask_logits,
    resolve_env_factory,
    shared_info_value,
    validate_parallel_env,
)
from .mappo import flatten_global_state
from .trajectory import (
    clipped_value_loss,
    compute_gae,
    episode_history_entry,
    normalize_advantage,
    ppo_approx_kl,
)


ALGORITHM_NAME = "Joint PPO"
DEFAULT_MAX_JOINT_ACTION_DIM = 100_000


class JointActor(nn.Module):
    joint_action_dim: int
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
            self.joint_action_dim,
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
    state: jax.Array


class JointPPOTrainState(NamedTuple):
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any


class JointPPOEnvSpec(NamedTuple):
    agent_ids: tuple[str, ...]
    action_dim: int
    state_dim: int
    joint_action_dim: int
    observation_spaces: tuple[gym.Space, ...]


def _build_joint_action_table(num_agents: int, action_dim: int) -> np.ndarray:
    joint_actions = tuple(product(range(int(action_dim)), repeat=int(num_agents)))
    return np.asarray(joint_actions, dtype=np.int32)


def _decode_joint_actions(
    joint_actions: jax.Array | np.ndarray,
    joint_action_table: jax.Array | np.ndarray,
) -> jax.Array:
    return jnp.take(
        jnp.asarray(joint_action_table, dtype=jnp.int32),
        jnp.asarray(joint_actions, dtype=jnp.int32),
        axis=0,
    )


def _joint_action_masks(
    action_masks: jax.Array | np.ndarray,
    joint_action_table: jax.Array | np.ndarray,
) -> jax.Array:
    mask = jnp.asarray(action_masks, dtype=bool)
    table = jnp.asarray(joint_action_table, dtype=jnp.int32)
    num_agents = table.shape[1]
    flat_mask = mask.reshape((-1, num_agents, mask.shape[-1]))
    agent_indices = jnp.arange(num_agents, dtype=jnp.int32)[:, None]
    table_by_agent = jnp.swapaxes(table, 0, 1)

    def _single_joint_mask(single_mask: jax.Array) -> jax.Array:
        return jnp.all(single_mask[agent_indices, table_by_agent], axis=0)

    flat_joint_mask = jax.vmap(_single_joint_mask)(flat_mask)
    return flat_joint_mask.reshape(mask.shape[:-2] + (table.shape[0],))


def joint_action_masks_from_infos(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    joint_action_dim: int,
) -> np.ndarray:
    resolved_mask: np.ndarray | None = None
    for agent in agent_ids:
        agent_info = infos.get(agent, {})
        raw_mask = agent_info.get("joint_action_mask")
        if raw_mask is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
        expected = (int(joint_action_dim),)
        if mask.shape != expected:
            raise ValueError(
                f"joint_action_mask for {agent!r} has shape {mask.shape}, "
                f"expected {expected}."
            )
        if resolved_mask is None:
            resolved_mask = mask
        elif not np.array_equal(resolved_mask, mask):
            raise ValueError("All agents must report the same joint_action_mask.")
    if resolved_mask is None:
        return np.ones(int(joint_action_dim), dtype=bool)
    if not bool(resolved_mask.any()):
        raise RuntimeError("joint_action_mask contains no valid joint actions.")
    return resolved_mask.astype(bool, copy=True)


def _effective_joint_action_masks(
    action_masks: jax.Array | np.ndarray,
    joint_action_masks: jax.Array | np.ndarray,
    joint_action_table: jax.Array | np.ndarray,
) -> jax.Array:
    cartesian_mask = _joint_action_masks(action_masks, joint_action_table)
    provided_mask = jnp.asarray(joint_action_masks, dtype=bool)
    return jnp.logical_and(cartesian_mask, provided_mask)


def assert_joint_actions_respect_masks(
    joint_action: int,
    joint_action_mask: np.ndarray,
    *,
    phase: str,
    env_index: int,
    global_step: int,
    update_idx: int,
) -> None:
    mask = np.asarray(joint_action_mask, dtype=bool).reshape(-1)
    action = int(joint_action)
    if 0 <= action < int(mask.shape[0]) and bool(mask[action]):
        return
    allowed = np.flatnonzero(mask).astype(int).tolist()
    raise AssertionError(
        f"{phase} selected invalid joint action {action} for env {env_index}; "
        f"allowed joint actions {allowed}. global_step={global_step}, "
        f"update_idx={update_idx}."
    )


def validate_joint_ppo_env(
    env: ParallelEnv,
    *,
    max_joint_action_dim: int = DEFAULT_MAX_JOINT_ACTION_DIM,
) -> JointPPOEnvSpec:
    base_spec = validate_parallel_env(env)
    joint_action_dim = int(base_spec.action_dim) ** len(base_spec.agent_ids)
    if joint_action_dim > int(max_joint_action_dim):
        raise ValueError(
            "Joint PPO joint action space has "
            f"{joint_action_dim} actions, exceeding MAX_JOINT_ACTION_DIM="
            f"{int(max_joint_action_dim)}."
        )
    env.reset(seed=0)
    state = flatten_global_state(env, algorithm_name=ALGORITHM_NAME)
    return JointPPOEnvSpec(
        agent_ids=base_spec.agent_ids,
        action_dim=base_spec.action_dim,
        state_dim=int(state.shape[0]),
        joint_action_dim=joint_action_dim,
        observation_spaces=base_spec.observation_spaces,
    )


def init_train_state(
    rng: jax.Array,
    actor: JointActor,
    critic: CentralCritic,
    tx: optax.GradientTransformation,
    *,
    state_dim: int,
) -> JointPPOTrainState:
    init_state = jnp.zeros((state_dim,), dtype=jnp.float32)
    actor_rng, critic_rng = jax.random.split(rng)
    actor_params = actor.init(actor_rng, init_state)
    critic_params = critic.init(critic_rng, init_state)
    actor_opt_state = tx.init(actor_params)
    critic_opt_state = tx.init(critic_params)
    return JointPPOTrainState(
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
        env_spec = validate_joint_ppo_env(
            probe_env,
            max_joint_action_dim=int(
                config.get("MAX_JOINT_ACTION_DIM", DEFAULT_MAX_JOINT_ACTION_DIM)
            ),
        )
    finally:
        probe_env.close()

    config.setdefault("CLIP_VALUE_LOSS", True)
    config.setdefault("VALUE_CLIP_EPS", config["CLIP_EPS"])

    joint_action_table = _build_joint_action_table(
        len(env_spec.agent_ids),
        env_spec.action_dim,
    )
    joint_action_table_jax = jnp.asarray(joint_action_table, dtype=jnp.int32)
    actor = JointActor(
        env_spec.joint_action_dim,
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

    def apply_actor(params: Any, state: jax.Array) -> jax.Array:
        return actor.apply(params, state)

    def apply_critic(params: Any, state: jax.Array) -> jax.Array:
        return critic.apply(params, state)

    sample_action_jit = jax.jit(
        lambda actor_params, critic_params, state, action_mask, joint_action_mask, rng: _sample_actions(
            apply_actor,
            apply_critic,
            actor_params,
            critic_params,
            state,
            action_mask,
            joint_action_mask,
            joint_action_table_jax,
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
                state_dim=env_spec.state_dim,
            )

            last_state = np.zeros((num_envs, env_spec.state_dim), dtype=np.float32)
            last_action_masks = all_true_action_masks(
                num_envs,
                num_agents,
                env_spec.action_dim,
            )
            last_joint_action_masks = np.ones(
                (num_envs, env_spec.joint_action_dim),
                dtype=bool,
            )
            for env_idx, env in enumerate(envs):
                _, infos = env.reset(seed=int(config.get("SEED", 0)) + reset_counter)
                reset_counter += 1
                last_action_masks[env_idx] = action_masks_from_infos(
                    infos,
                    env_spec.agent_ids,
                    env_spec.action_dim,
                )
                last_joint_action_masks[env_idx] = joint_action_masks_from_infos(
                    infos,
                    env_spec.agent_ids,
                    env_spec.joint_action_dim,
                )
                last_state[env_idx] = flatten_global_state(
                    env,
                    expected_dim=env_spec.state_dim,
                    validate_state_space=False,
                    algorithm_name=ALGORITHM_NAME,
                )

            for update_idx in range(int(config["NUM_UPDATES"])):
                traj_states = np.zeros(
                    (num_steps, num_envs, env_spec.state_dim),
                    dtype=np.float32,
                )
                traj_joint_actions = np.zeros((num_steps, num_envs), dtype=np.int32)
                traj_joint_action_masks = np.zeros(
                    (num_steps, num_envs, env_spec.joint_action_dim),
                    dtype=bool,
                )
                traj_log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_values = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_next_values = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_team_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_agent_rewards = np.zeros(
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
                traj_terminated = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_episode_dones = np.zeros((num_steps, num_envs), dtype=np.float32)
                completed_returns: list[np.ndarray] = []
                completed_lengths: list[int] = []
                completed_safety_violation_counts: list[float] = []

                for step_idx in range(num_steps):
                    state_batch = jnp.asarray(last_state)
                    action_mask_batch = jnp.asarray(last_action_masks)
                    joint_action_mask_batch = jnp.asarray(last_joint_action_masks)
                    rng, sample_rng = jax.random.split(rng)
                    (
                        actions,
                        joint_actions,
                        log_probs,
                        values,
                        joint_action_masks,
                    ) = sample_action_jit(
                        train_state.actor_params,
                        train_state.critic_params,
                        state_batch,
                        action_mask_batch,
                        joint_action_mask_batch,
                        sample_rng,
                    )
                    actions_np = np.asarray(actions)
                    joint_actions_np = np.asarray(joint_actions)
                    log_probs_np = np.asarray(log_probs)
                    values_np = np.asarray(values)

                    next_state = np.zeros_like(last_state)
                    next_state_for_value = np.zeros_like(last_state)
                    reward_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    terminated_batch = np.zeros(num_envs, dtype=np.float32)
                    episode_done_batch = np.zeros(num_envs, dtype=np.float32)

                    for env_idx, env in enumerate(envs):
                        global_step = (
                            update_idx * batch_size
                            + step_idx * num_envs
                            + env_idx
                            + 1
                        )
                        assert_actions_respect_masks(
                            actions_np[env_idx],
                            last_action_masks[env_idx],
                            env_spec.agent_ids,
                            phase="Joint PPO rollout sampling",
                            env_index=env_idx,
                            global_step=global_step,
                            update_idx=update_idx,
                        )
                        assert_joint_actions_respect_masks(
                            joint_actions_np[env_idx],
                            np.asarray(joint_action_masks)[env_idx],
                            phase="Joint PPO rollout sampling",
                            env_index=env_idx,
                            global_step=global_step,
                            update_idx=update_idx,
                        )
                        action_dict = actions_array_to_dict(
                            actions_np[env_idx],
                            env_spec.agent_ids,
                        )
                        (
                            _obs_dict,
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
                                "Joint PPO expects all agents in an environment to "
                                "finish together."
                            )
                        if not np.all(terminated_array == terminated_array[0]):
                            raise ValueError(
                                "Joint PPO expects all agents in an environment to "
                                "terminate together."
                            )

                        reward_batch[env_idx] = reward_array
                        terminated_batch[env_idx] = terminated_array[0]
                        episode_done_batch[env_idx] = done_array[0]
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
                        next_state_for_value[env_idx] = flatten_global_state(
                            env,
                            expected_dim=env_spec.state_dim,
                            validate_state_space=False,
                            algorithm_name=ALGORITHM_NAME,
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
                                    t_end=global_step,
                                    ep_len=int(episode_lengths[env_idx]),
                                    returns=episode_returns[env_idx],
                                    agent_ids=env_spec.agent_ids,
                                    safety_violations=episode_safety_violations,
                                )
                            )
                            total_completed_episodes += 1
                            _, infos = env.reset(
                                seed=int(config.get("SEED", 0)) + reset_counter
                            )
                            reset_counter += 1
                            episode_returns[env_idx].fill(0.0)
                            episode_lengths[env_idx] = 0
                            episode_start_steps[env_idx] = global_step

                        last_action_masks[env_idx] = action_masks_from_infos(
                            infos,
                            env_spec.agent_ids,
                            env_spec.action_dim,
                        )
                        last_joint_action_masks[env_idx] = joint_action_masks_from_infos(
                            infos,
                            env_spec.agent_ids,
                            env_spec.joint_action_dim,
                        )
                        next_state[env_idx] = flatten_global_state(
                            env,
                            expected_dim=env_spec.state_dim,
                            validate_state_space=False,
                            algorithm_name=ALGORITHM_NAME,
                        )

                    traj_states[step_idx] = last_state
                    traj_joint_actions[step_idx] = joint_actions_np
                    traj_joint_action_masks[step_idx] = np.asarray(joint_action_masks)
                    traj_log_probs[step_idx] = log_probs_np
                    traj_values[step_idx] = values_np
                    traj_next_values[step_idx] = np.asarray(
                        last_value_jit(
                            train_state.critic_params,
                            jnp.asarray(next_state_for_value),
                        )
                    )
                    traj_agent_rewards[step_idx] = reward_batch
                    traj_team_rewards[step_idx] = reward_batch.mean(axis=1)
                    traj_terminated[step_idx] = terminated_batch
                    traj_episode_dones[step_idx] = episode_done_batch
                    last_state = next_state

                traj_batch = Transition(
                    terminated=jnp.asarray(traj_terminated),
                    episode_done=jnp.asarray(traj_episode_dones),
                    action=jnp.asarray(traj_joint_actions),
                    action_mask=jnp.asarray(traj_joint_action_masks),
                    value=jnp.asarray(traj_values),
                    next_value=jnp.asarray(traj_next_values),
                    reward=jnp.asarray(traj_team_rewards),
                    log_prob=jnp.asarray(traj_log_probs),
                    state=jnp.asarray(traj_states),
                )

                advantages, targets = compute_gae_jit(traj_batch)
                flat_batch = {
                    "state": traj_batch.state.reshape(batch_size, env_spec.state_dim),
                    "action": traj_batch.action.reshape(batch_size),
                    "action_mask": traj_batch.action_mask.reshape(
                        batch_size,
                        env_spec.joint_action_dim,
                    ),
                    "log_prob": traj_batch.log_prob.reshape(batch_size),
                    "value": traj_batch.value.reshape(batch_size),
                    "advantage": advantages.reshape(batch_size),
                    "target": targets.reshape(batch_size),
                }

                minibatch_metrics: list[dict[str, np.ndarray]] = []
                for _ in range(int(config["UPDATE_EPOCHS"])):
                    rng, perm_rng = jax.random.split(rng)
                    permutation = jax.random.permutation(perm_rng, batch_size)
                    shuffled = {
                        key: jnp.take(value, permutation, axis=0)
                        for key, value in flat_batch.items()
                    }
                    for batch_start in range(0, batch_size, minibatch_size):
                        minibatch = {
                            key: value[batch_start : batch_start + minibatch_size]
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

                step_reward_mean = traj_agent_rewards.mean(axis=(0, 1))
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
                    "joint_action_dim": float(env_spec.joint_action_dim),
                }

                for key, value in aggregated_loss_metrics.items():
                    metrics[key] = float(value.mean())
                for agent_idx, agent_id in enumerate(env_spec.agent_ids):
                    metrics[f"{agent_id}/reward_mean"] = float(step_reward_mean[agent_idx])
                    metrics[f"{agent_id}/episode_return"] = float(
                        mean_episode_return[agent_idx]
                    )

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
                "joint_action_dim": env_spec.joint_action_dim,
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
    state: jax.Array,
    action_mask: jax.Array,
    joint_action_mask: jax.Array,
    joint_action_table: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    logits = apply_actor(actor_params, state)
    effective_joint_action_mask = _effective_joint_action_masks(
        action_mask,
        joint_action_mask,
        joint_action_table,
    )
    logits = mask_logits(logits, effective_joint_action_mask)
    values = apply_critic(critic_params, state)
    pi = distrax.Categorical(logits=logits)
    joint_actions = pi.sample(seed=rng)
    log_probs = pi.log_prob(joint_actions)
    actions = _decode_joint_actions(joint_actions, joint_action_table)
    return actions, joint_actions, log_probs, values, effective_joint_action_mask


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
    actor: JointActor,
    critic: CentralCritic,
    tx: optax.GradientTransformation,
    train_state: JointPPOTrainState,
    minibatch: Mapping[str, jax.Array],
) -> tuple[JointPPOTrainState, dict[str, jax.Array]]:
    clip_eps = jnp.asarray(config["CLIP_EPS"], dtype=jnp.float32)
    value_clip_eps = jnp.asarray(config["VALUE_CLIP_EPS"], dtype=jnp.float32)
    ent_coef = jnp.asarray(config["ENT_COEF"], dtype=jnp.float32)
    vf_coef = jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    clip_value_loss = bool(config.get("CLIP_VALUE_LOSS", True))

    def _loss_fn(
        actor_params: Any,
        critic_params: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = actor.apply(actor_params, minibatch["state"])
        logits = mask_logits(logits, minibatch["action_mask"])
        value = critic.apply(critic_params, minibatch["state"])
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(minibatch["action"])

        value_loss = clipped_value_loss(
            value=value,
            old_value=minibatch["value"],
            target=minibatch["target"],
            clip_eps=value_clip_eps,
            clip_value_loss=clip_value_loss,
        )

        normalized_advantage = normalize_advantage(minibatch["advantage"])
        log_ratio = log_prob - minibatch["log_prob"]
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

    (loss, loss_info), (actor_grads, critic_grads) = jax.value_and_grad(
        _loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(train_state.actor_params, train_state.critic_params)

    actor_updates, next_actor_opt_state = tx.update(
        actor_grads,
        train_state.actor_opt_state,
        train_state.actor_params,
    )
    critic_updates, next_critic_opt_state = tx.update(
        critic_grads,
        train_state.critic_opt_state,
        train_state.critic_params,
    )
    next_actor_params = optax.apply_updates(train_state.actor_params, actor_updates)
    next_critic_params = optax.apply_updates(train_state.critic_params, critic_updates)

    metrics = {key: value for key, value in loss_info.items()}
    metrics["total_loss"] = loss

    return (
        JointPPOTrainState(
            actor_params=next_actor_params,
            critic_params=next_critic_params,
            actor_opt_state=next_actor_opt_state,
            critic_opt_state=next_critic_opt_state,
        ),
        metrics,
    )

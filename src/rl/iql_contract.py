from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.experiments.safety_runs import (
    build_candidate_contract_library_template,
    build_contract_library_template,
    make_contract_env_factory,
    serialize_contract_library_template,
)
from src.shield import (
    ContractSynthesisConfig,
    DiscountedUCBProfileSelector,
    VotingConfig,
)

from .ippo import (
    EnvFactory,
    ProgressCallback,
    assert_actions_respect_masks,
    actions_array_to_dict,
    action_masks_from_infos,
    all_true_action_masks,
    dict_observations_to_array,
    dict_values_to_array,
    resolve_env_factory,
    shared_info_value,
    validate_parallel_env,
)
from .ippo_contract import (
    _contract_reset_options,
    _record_contract_event,
    _record_contract_undercoverage_event,
    _team_block_score,
)
from .iql import (
    IQLTrainState,
    QNetwork,
    _compute_num_updates,
    _sample_actions,
    _update_minibatch,
    _validate_iql_config,
    add_to_replay_buffer,
    init_replay_buffer,
    init_train_state,
    sample_replay_buffer,
)
from .trajectory import episode_history_entry


def make_train(
    config: Mapping[str, Any],
    env_factory: EnvFactory | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[jax.Array], dict[str, Any]]:
    algorithm_name = "Contract-IQL"
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
    for required_key in (
        "CONTRACT_MODEL_FACTORY",
        "CONTRACT_GLOBAL_FORMULA",
        "CONTRACT_GLOBAL_FORMULA_NAME",
    ):
        if required_key not in config:
            raise ValueError(f"Contract-IQL requires {required_key}.")

    resolved_env_factory = resolve_env_factory(factory_ref)
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    contract_candidate_labels = config.get("CONTRACT_CANDIDATE_LABELS")
    synthesis_config = ContractSynthesisConfig(
        depth=int(config.get("CONTRACT_DEPTH", 1)),
        max_candidates=int(config.get("CONTRACT_MAX_CANDIDATES", 64)),
        max_profiles=int(config.get("CONTRACT_MAX_PROFILES", 64)),
        max_active_per_agent=int(config.get("CONTRACT_MAX_ACTIVE_PER_AGENT", 1)),
        max_atomic_props=int(config.get("CONTRACT_MAX_ATOMIC_PROPS", 16)),
        max_states=config.get("CONTRACT_MAX_STATES"),
        reuse_certification_caches=bool(
            config.get("CONTRACT_REUSE_CERTIFICATION_CACHES", True)
        ),
        cache_certification_successors=bool(
            config.get("CONTRACT_CACHE_CERTIFICATION_SUCCESSORS", True)
        ),
        include_weak_until=bool(config.get("CONTRACT_INCLUDE_WEAK_UNTIL", False)),
        print_candidates=bool(config.get("CONTRACT_PRINT_CANDIDATES", False)),
        candidate_labels=(
            tuple(str(label) for label in contract_candidate_labels)
            if contract_candidate_labels is not None
            else None
        ),
        use_model_local_alphabet=bool(
            config.get("CONTRACT_USE_MODEL_LOCAL_ALPHABET", True)
        ),
        use_temporal_form_heuristic=bool(
            config.get("CONTRACT_USE_TEMPORAL_FORM_HEURISTIC", True)
        ),
        use_model_seed_formulas=bool(
            config.get("CONTRACT_USE_MODEL_SEED_FORMULAS", False)
        ),
        prune_equivalent_candidates=bool(
            config.get("CONTRACT_PRUNE_EQUIVALENT_CANDIDATES", True)
        ),
        prune_equivalent_profiles=bool(
            config.get("CONTRACT_PRUNE_EQUIVALENT_PROFILES", True)
        ),
    )
    if bool(config.get("CONTRACT_USE_CANDIDATE_PROFILES", False)):
        contract_template = build_candidate_contract_library_template(
            env_factory=resolved_env_factory,
            env_kwargs=env_kwargs,
            model_factory=config["CONTRACT_MODEL_FACTORY"],
            global_formula_name=str(config["CONTRACT_GLOBAL_FORMULA_NAME"]),
            global_formula=str(config["CONTRACT_GLOBAL_FORMULA"]),
            profile_specs=config.get("CONTRACT_CANDIDATE_PROFILES"),
            profile_factory=config.get("CONTRACT_CANDIDATE_PROFILE_FACTORY"),
            max_states=config.get("CONTRACT_MAX_STATES"),
            reuse_certification_caches=bool(
                config.get("CONTRACT_REUSE_CERTIFICATION_CACHES", True)
            ),
            cache_certification_successors=bool(
                config.get("CONTRACT_CACHE_CERTIFICATION_SUCCESSORS", True)
            ),
            synthesis_seed=int(config.get("SEED", 0)),
        )
    else:
        contract_template = build_contract_library_template(
            env_factory=resolved_env_factory,
            env_kwargs=env_kwargs,
            model_factory=config["CONTRACT_MODEL_FACTORY"],
            global_formula_name=str(config["CONTRACT_GLOBAL_FORMULA_NAME"]),
            global_formula=str(config["CONTRACT_GLOBAL_FORMULA"]),
            config=synthesis_config,
            synthesis_seed=int(config.get("SEED", 0)),
        )
    contract_env_factory = make_contract_env_factory(
        base_env_factory=resolved_env_factory,
        model_factory=config["CONTRACT_MODEL_FACTORY"],
        contract_template=contract_template,
    )

    probe_env = contract_env_factory(**env_kwargs)
    try:
        env_spec = validate_parallel_env(probe_env)
    finally:
        probe_env.close()

    profile_ids = tuple(
        certified.profile.profile_id
        for certified in contract_template.library.certified_profiles
    )
    initial_profile_id = contract_template.library.initial_profile.profile.profile_id
    voting_config = VotingConfig(
        warmup_episodes=int(config.get("CONTRACT_WARMUP_EPISODES", 0)),
        dwell_episodes=max(int(config.get("CONTRACT_DWELL_EPISODES", 10)), 1),
        bandit_discount=float(config.get("CONTRACT_BANDIT_DISCOUNT", 0.95)),
        bandit_exploration_coef=float(
            config.get("CONTRACT_BANDIT_EXPLORATION_COEF", 1.0)
        ),
    )
    profile_selector = DiscountedUCBProfileSelector(profile_ids, voting_config)

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

        envs = [contract_env_factory(**env_kwargs) for _ in range(num_envs)]
        episode_returns = np.zeros((num_envs, num_agents), dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        episode_start_steps = np.zeros(num_envs, dtype=np.int64)
        reset_counter = 0
        total_completed_episodes = 0
        active_profile_id = initial_profile_id
        last_vote_winner = active_profile_id
        last_vote_margin = 0
        block_returns: list[np.ndarray] = []
        metrics_history: dict[str, list[float]] = defaultdict(list)
        episode_history: list[dict[str, object]] = []
        contract_events: list[dict[str, Any]] = []
        vote_events: list[dict[str, Any]] = []
        contract_undercoverage_events: list[dict[str, Any]] = []

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
                obs_dict, infos = env.reset(
                    seed=int(config.get("SEED", 0)) + reset_counter,
                    options=_contract_reset_options(
                        completed_episodes=total_completed_episodes,
                        global_step=0,
                        profile_id=active_profile_id,
                        vote_winner=last_vote_winner,
                        vote_margin=last_vote_margin,
                    ),
                )
                reset_counter += 1
                maybe_event = _record_contract_event(
                    infos,
                    env_spec.agent_ids,
                    step=0,
                    completed_episodes=total_completed_episodes,
                )
                if maybe_event is not None:
                    contract_events.append(maybe_event)
                maybe_undercoverage_event = _record_contract_undercoverage_event(
                    infos,
                    env_spec.agent_ids,
                    step=0,
                    completed_episodes=total_completed_episodes,
                )
                if maybe_undercoverage_event is not None:
                    contract_undercoverage_events.append(maybe_undercoverage_event)
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
                traj_profile_indices = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_profile_changed = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_vote_margins = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_permissiveness = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_contract_shield_expanded = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                latest_contract_shield_expansions = np.zeros(
                    num_envs,
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
                traj_load_attempt_counts = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_load_failure_counts = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                traj_load_success_counts = np.zeros(
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
                    current_global_step = (
                        update_idx * batch_size + (step_idx + 1) * num_envs
                    )

                    for env_idx, env in enumerate(envs):
                        env_global_step = current_global_step - num_envs + env_idx + 1
                        assert_actions_respect_masks(
                            actions_np[env_idx],
                            last_action_masks[env_idx],
                            env_spec.agent_ids,
                            phase="Contract-IQL rollout sampling",
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
                                "Contract-IQL expects all agents in an environment "
                                "to finish together."
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
                        traj_load_attempt_counts[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "load_attempts_mean",
                            default=0.0,
                        )
                        traj_load_failure_counts[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "load_failures_mean",
                            default=0.0,
                        )
                        traj_load_success_counts[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "load_successes_mean",
                            default=0.0,
                        )
                        traj_profile_indices[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_profile_index",
                            default=0.0,
                        )
                        traj_profile_changed[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_profile_changed",
                            default=0.0,
                        )
                        traj_vote_margins[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_vote_margin",
                            default=0.0,
                        )
                        traj_permissiveness[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_permissiveness",
                            default=0.0,
                        )
                        traj_contract_shield_expanded[step_idx, env_idx] = (
                            shared_info_value(
                                infos,
                                env_spec.agent_ids,
                                "contract_shield_expanded",
                                default=0.0,
                            )
                        )
                        latest_contract_shield_expansions[env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_shield_expansions_cumulative",
                            default=0.0,
                        )
                        maybe_undercoverage_event = _record_contract_undercoverage_event(
                            infos,
                            env_spec.agent_ids,
                            step=current_global_step,
                            completed_episodes=total_completed_episodes,
                        )
                        if maybe_undercoverage_event is not None:
                            contract_undercoverage_events.append(
                                maybe_undercoverage_event
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
                            if total_completed_episodes > voting_config.warmup_episodes:
                                block_returns.append(episode_returns[env_idx].copy())

                            if len(block_returns) >= voting_config.dwell_episodes:
                                (
                                    observed_score,
                                    observed_team_return,
                                ) = _team_block_score(block_returns)
                                previous_profile_id = active_profile_id
                                profile_selector.record_block(
                                    active_profile_id,
                                    observed_score,
                                )
                                bandit_result = profile_selector.select(
                                    current_profile_id=active_profile_id
                                )
                                active_profile_id = bandit_result.winner_profile_id
                                last_vote_winner = bandit_result.winner_profile_id
                                last_vote_margin = 0
                                selector_event = {
                                    "selector": "bandit",
                                    "objective": "team-return",
                                    "votes": {},
                                    "vote_counts": {},
                                    "margin": 0,
                                    "arm_means": dict(bandit_result.arm_means),
                                    "arm_counts": dict(bandit_result.arm_counts),
                                    "arm_ucb_scores": dict(
                                        bandit_result.arm_ucb_scores
                                    ),
                                    "selection_margin": float(
                                        bandit_result.selection_margin
                                    ),
                                }
                                vote_events.append(
                                    {
                                        "step": int(current_global_step),
                                        "completed_episodes": int(
                                            total_completed_episodes
                                        ),
                                        "previous_profile_id": previous_profile_id,
                                        "winner_profile_id": active_profile_id,
                                        "observed_score": float(observed_score),
                                        "observed_team_return": float(
                                            observed_team_return
                                        ),
                                        **selector_event,
                                    }
                                )
                                block_returns.clear()

                            obs_dict, reset_infos = env.reset(
                                seed=int(config.get("SEED", 0)) + reset_counter,
                                options=_contract_reset_options(
                                    completed_episodes=total_completed_episodes,
                                    global_step=current_global_step,
                                    profile_id=active_profile_id,
                                    vote_winner=last_vote_winner,
                                    vote_margin=last_vote_margin,
                                ),
                            )
                            reset_counter += 1
                            maybe_event = _record_contract_event(
                                reset_infos,
                                env_spec.agent_ids,
                                step=current_global_step,
                                completed_episodes=total_completed_episodes,
                            )
                            if maybe_event is not None:
                                contract_events.append(maybe_event)
                            episode_returns[env_idx].fill(0.0)
                            episode_lengths[env_idx] = 0
                            episode_start_steps[env_idx] = env_global_step
                            infos = reset_infos

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
                    aggregated_loss_metrics = {
                        "q_loss": np.zeros(num_agents, dtype=np.float32),
                        "td_error_mean": np.zeros(num_agents, dtype=np.float32),
                        "q_value_mean": traj_action_q_values.mean(axis=(0, 1)),
                    }

                step_reward_mean = traj_rewards.mean(axis=(0, 1))
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
                    "safety_violations_mean": float(
                        traj_safety_violation_counts.mean()
                    ),
                    "safety_violations_agent_fraction_mean": float(
                        traj_safety_violation_fractions.mean()
                    ),
                    "safety_violations_cumulative": float(
                        latest_cumulative_safety_violations.sum()
                    ),
                    "episode_safety_violations_mean": mean_episode_safety_violations,
                    "shield_interventions_mean": float(
                        traj_shield_intervention_counts.mean()
                    ),
                    "shield_interventions_agent_fraction_mean": float(
                        traj_shield_intervention_fractions.mean()
                    ),
                    "load_attempts_mean": float(traj_load_attempt_counts.mean()),
                    "load_failures_mean": float(traj_load_failure_counts.mean()),
                    "load_successes_mean": float(traj_load_success_counts.mean()),
                    "contract_profile_index": float(traj_profile_indices.mean()),
                    "contract_profile_changed": float(traj_profile_changed.max()),
                    "contract_vote_margin": float(traj_vote_margins.mean()),
                    "contract_permissiveness": float(traj_permissiveness.mean()),
                    "contract_shield_expanded": float(
                        traj_contract_shield_expanded.max()
                    ),
                    "contract_shield_expansions_cumulative": float(
                        latest_contract_shield_expansions.sum()
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
                        f"contract_profile={metrics['contract_profile_index']:.1f}"
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

            serialized_contract_library = serialize_contract_library_template(contract_template)
            return {
                "train_state": train_state,
                "agent_ids": env_spec.agent_ids,
                "metrics": {
                    key: jnp.asarray(values)
                    for key, values in metrics_history.items()
                },
                "contract_library": serialized_contract_library,
                "contract_events": contract_events,
                "contract_vote_events": vote_events,
                "contract_undercoverage_events": contract_undercoverage_events,
                "episode_history": episode_history,
            }
        finally:
            for env in envs:
                env.close()

    return train

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
    ActorCritic,
    EnvFactory,
    ProgressCallback,
    Transition,
    _compute_gae,
    _compute_last_values,
    _sample_actions,
    _update_minibatch,
    assert_actions_respect_masks,
    actions_array_to_dict,
    action_masks_from_infos,
    all_true_action_masks,
    dict_observations_to_array,
    dict_values_to_array,
    init_train_state,
    resolve_env_factory,
    shared_info_value,
    validate_parallel_env,
)
from .trajectory import episode_history_entry


def _record_contract_event(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    *,
    step: int,
    completed_episodes: int,
) -> dict[str, Any] | None:
    if not any(
        bool(infos.get(agent_id, {}).get("contract_profile_changed", False))
        for agent_id in agent_ids
    ):
        return None
    first_agent = agent_ids[0]
    profile_id = str(infos[first_agent]["contract_profile_id"])
    return {
        "step": int(step),
        "completed_episodes": int(completed_episodes),
        "profile_id": profile_id,
        "profile_index": int(infos[first_agent]["contract_profile_index"]),
        "label": profile_id,
        "legend_label": "Contract change",
        "kind": "dashed",
        "color": "#7c3aed",
        "formula": {
            agent_id: str(infos.get(agent_id, {}).get("contract_formula", "t"))
            for agent_id in agent_ids
        },
        "active_candidates": {
            agent_id: tuple(
                infos.get(agent_id, {}).get("contract_active_candidates", ())
            )
            for agent_id in agent_ids
        },
    }


def _record_contract_undercoverage_event(
    infos: Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
    *,
    step: int,
    completed_episodes: int,
) -> dict[str, Any] | None:
    for agent_id in agent_ids:
        raw_event = infos.get(agent_id, {}).get("contract_undercoverage_event")
        if raw_event is None:
            continue
        event = dict(raw_event)
        event["step"] = int(step)
        event["global_step"] = int(step)
        event["completed_episodes"] = int(completed_episodes)
        return event
    return None


def _contract_reset_options(
    *,
    completed_episodes: int,
    global_step: int,
    profile_id: str,
    vote_winner: str,
    vote_margin: int,
) -> dict[str, Any]:
    return {
        "completed_episodes": int(completed_episodes),
        "global_step": int(global_step),
        "contract_profile_id": str(profile_id),
        "contract_vote_winner": str(vote_winner),
        "contract_vote_margin": int(vote_margin),
    }


def _team_block_score(
    returns: list[np.ndarray],
) -> tuple[float, float]:
    mean_return = float(np.stack(returns, axis=0).mean()) if returns else 0.0
    score = mean_return
    return score, mean_return


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
    for required_key in (
        "CONTRACT_MODEL_FACTORY",
        "CONTRACT_GLOBAL_FORMULA",
        "CONTRACT_GLOBAL_FORMULA_NAME",
    ):
        if required_key not in config:
            raise ValueError(f"Contract-IPPO requires {required_key}.")

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
    compute_gae_jit = jax.jit(lambda traj_batch: _compute_gae(config, traj_batch))
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
                traj_obs = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.obs_dim),
                    dtype=np.float32,
                )
                traj_actions = np.zeros((num_steps, num_envs, num_agents), dtype=np.int32)
                traj_action_masks = np.zeros(
                    (num_steps, num_envs, num_agents, env_spec.action_dim),
                    dtype=bool,
                )
                traj_log_probs = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
                traj_values = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
                traj_next_values = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
                traj_rewards = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
                traj_safety_violation_counts = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_safety_violation_fractions = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_profile_indices = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_profile_changed = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_vote_margins = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_permissiveness = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_contract_shield_expanded = np.zeros(
                    (num_steps, num_envs),
                    dtype=np.float32,
                )
                latest_contract_shield_expansions = np.zeros(num_envs, dtype=np.float32)
                traj_shield_intervention_counts = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_shield_intervention_fractions = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_load_attempt_counts = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_load_failure_counts = np.zeros((num_steps, num_envs), dtype=np.float32)
                traj_load_success_counts = np.zeros((num_steps, num_envs), dtype=np.float32)
                latest_cumulative_safety_violations = np.zeros(num_envs, dtype=np.float32)
                traj_terminated = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
                traj_episode_dones = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
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
                    episode_done_batch = np.zeros((num_envs, num_agents), dtype=np.float32)
                    current_global_step = (
                        update_idx * batch_size + (step_idx + 1) * num_envs
                    )

                    for env_idx, env in enumerate(envs):
                        env_global_step = current_global_step - num_envs + env_idx + 1
                        assert_actions_respect_masks(
                            actions_np[env_idx],
                            last_action_masks[env_idx],
                            env_spec.agent_ids,
                            phase="Contract-IPPO rollout sampling",
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
                                "Contract-IPPO expects all agents in an environment to finish together."
                            )

                        reward_batch[env_idx] = reward_array
                        terminated_batch[env_idx] = terminated_array
                        episode_done_batch[env_idx] = done_array
                        traj_safety_violation_counts[step_idx, env_idx] = safety_violation_count
                        traj_safety_violation_fractions[step_idx, env_idx] = safety_violation_fraction
                        traj_shield_intervention_counts[step_idx, env_idx] = shield_intervention_count
                        traj_shield_intervention_fractions[step_idx, env_idx] = shield_intervention_fraction
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
                        traj_contract_shield_expanded[step_idx, env_idx] = shared_info_value(
                            infos,
                            env_spec.agent_ids,
                            "contract_shield_expanded",
                            default=0.0,
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
                            if total_completed_episodes > voting_config.warmup_episodes:
                                block_returns.append(episode_returns[env_idx].copy())

                            if (
                                len(block_returns) >= voting_config.dwell_episodes
                            ):
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
                                        "completed_episodes": int(total_completed_episodes),
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
                        train_state, loss_info = update_minibatch_jit(train_state, minibatch)
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
                    "safety_violations_mean": float(traj_safety_violation_counts.mean()),
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
                    for key, value in aggregated_loss_metrics.items():
                        metrics[f"{agent_id}/{key}"] = float(value[agent_idx])

                if config.get("DEBUG"):
                    print(
                        "update="
                        f"{update_idx} step={int(metrics['global_step'])} "
                        f"return={metrics['episode_return_mean']:.3f} "
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

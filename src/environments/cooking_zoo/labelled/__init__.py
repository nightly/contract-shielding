from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from itertools import product
from typing import Any

from ..impl.cooking_book.recipe import Recipe, RecipeNode
from ..impl.cooking_world.abstract_classes import (
    BlenderFood,
    ChopFood,
    DynamicObject,
    Food,
)
from ..impl.cooking_world.actions import ActionScheme1
from ..impl.cooking_world.world_objects import StringToClass

from src.shield.core import AbstractTransitionOutcome, Label, LocalAction


_WALK_ACTIONS = frozenset(ActionScheme1.WALK_ACTIONS)
_INTERACT_PRIMARY = int(ActionScheme1.INTERACT_PRIMARY)
_INTERACT_PICK_UP_SPECIAL = int(ActionScheme1.INTERACT_PICK_UP_SPECIAL)
_EXECUTE_ACTION = int(ActionScheme1.EXECUTE_ACTION)
_NOOP = int(ActionScheme1.NO_OP)
_STATION_KINDS = frozenset({"Deliversquare", "Cutboard"})
_RECIPE_CONTAINER_KINDS = frozenset({"Deliversquare", "Plate"})


def _kind_names_matching(base_class: type) -> frozenset[str]:
    return frozenset(
        kind
        for kind, cls in StringToClass.items()
        if isinstance(cls, type) and issubclass(cls, base_class)
    )


_BLENDER_FOOD_KINDS = _kind_names_matching(BlenderFood)
_CHOP_FOOD_KINDS = _kind_names_matching(ChopFood)
_DYNAMIC_KINDS = _kind_names_matching(DynamicObject)
_FOOD_KINDS = _kind_names_matching(Food)


COOKING_ZOO_PROTOCOL_LABEL_FAMILIES = (
    "recipe_role_ok",
    "delivery_station_ok",
    "cutboard_station_ok",
    "station_clearance_ok",
)


def cooking_zoo_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        str(agent_id): (
            *(f"{agent_id}_{family}" for family in COOKING_ZOO_PROTOCOL_LABEL_FAMILIES),
            f"{agent_id}_bad_delivery_attempt",
            f"{agent_id}_holding_teammate_only_ingredient",
            f"{agent_id}_blocking_delivery_access",
            f"{agent_id}_blocking_cutboard_access",
        )
        for agent_id in agent_ids
    }


def cooking_zoo_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    formula_props = tuple(
        prop
        for agent_id in agent_ids
        for prop in (
            f"{agent_id}_bad_delivery_attempt",
            f"{agent_id}_holding_teammate_only_ingredient",
            f"{agent_id}_blocking_delivery_access",
            f"{agent_id}_blocking_cutboard_access",
        )
    )
    shared_recipe_props = (
        "recipe_0_completed",
        "recipe_1_completed",
        "all_recipes_completed",
    )
    return {
        str(agent_id): (
            *(f"{agent_id}_{family}" for family in COOKING_ZOO_PROTOCOL_LABEL_FAMILIES),
            *formula_props,
            f"{agent_id}_holding_complete_dish",
            f"{agent_id}_holding_incomplete_dish",
            f"{agent_id}_holding_own_required_ingredient",
            f"{agent_id}_holding_teammate_requested_item",
            f"{agent_id}_ready_to_deliver",
            f"{agent_id}_ready_to_chop",
            *shared_recipe_props,
        )
        for agent_id in agent_ids
    }


@dataclass(frozen=True)
class CookingZooAgentState:
    location: tuple[int, int]
    orientation: int
    holding: int | None
    active: bool = True

    def __hash__(self) -> int:
        cached = self.__dict__.get("_cached_hash")
        if cached is None:
            cached = hash((self.location, self.orientation, self.holding, self.active))
            object.__setattr__(self, "_cached_hash", cached)
        return cached


@dataclass(frozen=True)
class CookingZooObjectState:
    object_id: int
    kind: str
    location: tuple[int, int]
    attrs: tuple[tuple[str, str], ...] = ()
    content: tuple[int, ...] = ()
    free: bool = True

    def __hash__(self) -> int:
        cached = self.__dict__.get("_cached_hash")
        if cached is None:
            cached = hash(
                (
                    self.object_id,
                    self.kind,
                    self.location,
                    self.attrs,
                    self.content,
                    self.free,
                )
            )
            object.__setattr__(self, "_cached_hash", cached)
        return cached


@dataclass(frozen=True)
class CookingZooState:
    agents: tuple[CookingZooAgentState, ...]
    objects: tuple[CookingZooObjectState, ...]
    recipe_completed: tuple[bool, ...]
    bad_delivery_attempts: frozenset[int] = frozenset()
    t: int = 0

    def __hash__(self) -> int:
        cached = self.__dict__.get("_cached_hash")
        if cached is None:
            cached = hash(
                (
                    self.agents,
                    self.objects,
                    self.recipe_completed,
                    self.bad_delivery_attempts,
                    self.t,
                )
            )
            object.__setattr__(self, "_cached_hash", cached)
        return cached


class CookingZooSafetyModel:
    def __init__(
        self,
        env,
        *,
        projection_mode: str = "object_delta",
        teammate_context_mode: str = "movement",
    ) -> None:
        if projection_mode not in {"object_delta", "role_core"}:
            raise ValueError(
                "CookingZooSafetyModel projection_mode must be one of "
                "'object_delta' or 'role_core'."
            )
        if teammate_context_mode not in {"movement", "enabled"}:
            raise ValueError(
                "CookingZooSafetyModel teammate_context_mode must be one of "
                "'movement' or 'enabled'."
            )
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.projection_mode = projection_mode
        self.teammate_context_mode = teammate_context_mode
        self.world = self.base_env.world
        self._validate_supported_config()
        self.agent_ids = tuple(self.base_env.possible_agents)
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._num_agents = len(self.agent_ids)
        self._actions = tuple(range(self.base_env.action_space(self.agent_ids[0]).n))
        self._movement_or_noop_actions = tuple(
            action
            for action in self._actions
            if action in _WALK_ACTIONS or int(action) == _NOOP
        )
        self._max_steps = int(getattr(self.base_env, "max_steps", 0) or 0)
        self._recipes = tuple(self.base_env.recipe_graphs)
        self._reward_scheme = dict(getattr(self.base_env, "reward_scheme", {}) or {})
        self._recipe_food_requirements = tuple(
            self._recipe_food_names(recipe)
            for recipe in self._recipes
        )
        self._role_core_attr_names = self._role_core_attribute_names()
        self._attrs_dict_cache: dict[
            tuple[tuple[str, str], ...],
            dict[str, str],
        ] = {}
        self._role_core_attrs_cache: dict[
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
        ] = {}
        self._role_core_content_token_cache: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            str,
        ] = {}
        self._safety_projection_cache: dict[CookingZooState, CookingZooState] = {}
        self._objects_by_id_cache: dict[
            CookingZooState,
            dict[int, CookingZooObjectState],
        ] = {}
        self._state_cache_limit = 8_192
        self._role_core_object_ids: dict[tuple[str, ...], int] = {}
        self._role_core_keys_by_object_id: dict[int, tuple[str, ...]] = {}
        self._recipe_relevant_food_kinds = frozenset().union(
            *self._recipe_food_requirements
        ) if self._recipe_food_requirements else frozenset()
        self._delivery_access_cells = self._station_access_cells("Deliversquare")
        self._cutboard_access_cells = self._station_access_cells("Cutboard")
        (
            self._station_access_projection,
            self._station_entry_projection,
        ) = self._station_projection_maps()
        initial_state = self.abstract_state(env)
        self._projection_objects = tuple(initial_state.objects)
        self._projection_dynamic_ids = frozenset(
            obj.object_id
            for obj in self._projection_objects
            if self._is_dynamic_kind(obj.kind)
        )
        self._projection_relevant_dynamic_ids = frozenset(
            obj.object_id
            for obj in self._projection_objects
            if self._is_projection_relevant_dynamic_kind(obj.kind)
        )
        self._projection_base_objects = tuple(
            obj
            for obj in self._projection_objects
            if (
                obj.object_id not in self._projection_dynamic_ids
                or obj.object_id in self._projection_relevant_dynamic_ids
            )
        )
        self._projection_base_objects_by_id = {
            obj.object_id: obj
            for obj in self._projection_base_objects
        }
        self._relevant_content_access_cells = self._content_access_cells(
            self._projection_base_objects
        )
        self._projection_initial_locations = tuple(
            agent.location
            for agent in initial_state.agents
        )
        self._projection_fallback_locations = self._safe_projection_fallback_locations(
            self._projection_initial_locations
        )
        self._projection_interesting_cells = frozenset(
            self._station_relevant_cells()
            | set(self._relevant_content_access_cells)
            | set(self._projection_fallback_locations)
        )
        self._orientation_relevant_cells = frozenset(
            set(self._delivery_access_cells)
            | set(self._cutboard_access_cells)
            | set(self._relevant_content_access_cells)
        )
        self._successor_cache: dict[
            tuple[CookingZooState, str, int],
            frozenset[CookingZooState],
        ] = {}

    def _validate_supported_config(self) -> None:
        if self.base_env.action_scheme != "scheme1":
            raise ValueError(
                "CookingZooSafetyModel exact successor enumeration supports only "
                "action_scheme='scheme1'."
            )
        if float(getattr(self.base_env, "agent_respawn_rate", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "CookingZooSafetyModel exact successor enumeration requires "
                "agent_respawn_rate=0.0."
            )
        if float(getattr(self.base_env, "agent_despawn_rate", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "CookingZooSafetyModel exact successor enumeration requires "
                "agent_despawn_rate=0.0."
            )
        reward_scheme = dict(getattr(self.base_env, "reward_scheme", {}) or {})
        if float(reward_scheme.get("recipe_node_reward", 0.0)) != 0.0:
            raise ValueError(
                "CookingZooSafetyModel exact reward outcomes currently require "
                "recipe_node_reward=0."
            )

    def initial_state(self, env: object) -> CookingZooState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> CookingZooState:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        world = base_env.world
        agent_object_ids = frozenset(
            int(agent.unique_id)
            for agent in world.agents[: self._num_agents]
            if hasattr(agent, "unique_id")
        )
        objects = tuple(
            sorted(
                (
                    self._without_agent_contents(
                        self._object_state(obj),
                        agent_object_ids,
                    )
                    for object_list in world.world_objects.values()
                    for obj in object_list
                ),
                key=lambda obj_state: obj_state.object_id,
            )
        )
        recipe_completed = tuple(bool(recipe.completed()) for recipe in base_env.recipe_graphs)
        agents = tuple(
            CookingZooAgentState(
                location=tuple(agent.location),
                orientation=int(agent.orientation),
                holding=(
                    int(agent.holding.unique_id)
                    if getattr(agent, "holding", None) is not None
                    else None
                ),
                active=bool(
                    world.active_agents[index]
                    if index < len(world.active_agents)
                    else True
                ),
            )
            for index, agent in enumerate(world.agents[: self._num_agents])
        )
        bad_delivery_attempts = frozenset(
            index
            for index, agent in enumerate(world.agents[: self._num_agents])
            if self._agent_has_bad_delivery_interaction(
                agent,
                index,
                objects,
                recipe_completed,
            )
        )
        return CookingZooState(
            agents=agents,
            objects=objects,
            recipe_completed=recipe_completed,
            bad_delivery_attempts=bad_delivery_attempts,
            t=int(getattr(base_env, "t", 0) or 0),
        )

    def safety_projection(self, state: CookingZooState) -> CookingZooState:
        cached_projection = self._safety_projection_cache.get(state)
        if cached_projection is not None:
            return cached_projection
        state_objects = self._objects_by_id(state)
        base_objects = self._projection_base_objects_by_id
        agents: list[CookingZooAgentState] = []
        held_ids: set[int] = set()
        for agent in state.agents:
            holding = agent.holding if agent.holding in state_objects else None
            orientation = int(agent.orientation)
            if self.projection_mode == "role_core":
                orientation = self._project_orientation(agent.location, orientation)
            agents.append(
                replace(
                    agent,
                    orientation=orientation,
                    holding=holding,
                    active=True,
                )
            )
            if holding is None:
                continue
            held_ids.add(holding)

        content_owner = {
            content_id: obj
            for obj in state_objects.values()
            for content_id in obj.content
        }
        base_content_owner = {
            content_id: obj
            for obj in base_objects.values()
            for content_id in obj.content
        }

        selected_static_ids: set[int] = set()
        selected_dynamic_ids: set[int] = set(held_ids)

        for obj_id in held_ids:
            owner = content_owner.get(obj_id)
            if owner is not None:
                selected_static_ids.add(owner.object_id)

        for obj in state_objects.values():
            if self._is_dynamic_kind(obj.kind):
                continue
            base_obj = base_objects.get(obj.object_id)
            if self._station_state_changed(obj, base_obj):
                selected_static_ids.add(obj.object_id)
                continue
            if self._has_projection_relevant_content(obj.content):
                selected_static_ids.add(obj.object_id)
                continue

        for static_id in tuple(selected_static_ids):
            static = state_objects.get(static_id)
            if static is None:
                continue
            selected_dynamic_ids.update(
                content_id
                for content_id in static.content
                if content_id in state_objects
                and self._is_projection_relevant_dynamic_kind(
                    state_objects[content_id].kind
                )
            )

        for obj_id in tuple(selected_dynamic_ids):
            base_owner = base_content_owner.get(obj_id)
            current_owner = content_owner.get(obj_id)
            if base_owner is None:
                continue
            if current_owner is None or current_owner.object_id != base_owner.object_id:
                selected_static_ids.add(base_owner.object_id)

        pending_dynamic_ids = list(selected_dynamic_ids)
        while pending_dynamic_ids:
            obj_id = pending_dynamic_ids.pop()
            obj = state_objects.get(obj_id)
            if obj is None:
                continue
            for content_id in obj.content:
                content = state_objects.get(content_id)
                if content is None or not self._is_projection_relevant_dynamic_kind(
                    content.kind
                ):
                    continue
                if content_id in selected_dynamic_ids:
                    continue
                selected_dynamic_ids.add(content_id)
                pending_dynamic_ids.append(content_id)

        objects: dict[int, CookingZooObjectState] = {}
        for static_id in sorted(selected_static_ids):
            static = state_objects.get(static_id)
            if static is None:
                continue
            filtered_content = tuple(
                content_id
                for content_id in static.content
                if content_id not in held_ids
            )
            if filtered_content != static.content:
                static = replace(static, content=filtered_content)
            base_obj = base_objects.get(static_id)
            if static == base_obj:
                continue
            objects[static_id] = static

        for agent_idx, agent in enumerate(agents):
            holding = agent.holding
            if holding is None:
                continue
            held = state_objects[holding]
            objects[holding] = replace(
                held,
                location=agent.location,
                free=False,
            )

        for obj_id in sorted(selected_dynamic_ids - held_ids):
            obj = state_objects.get(obj_id)
            if obj is None:
                continue
            owner = content_owner.get(obj_id)
            location = owner.location if owner is not None else obj.location
            objects[obj_id] = replace(
                obj,
                location=location,
                free=bool(owner is None and obj.free),
            )

        projected = CookingZooState(
            agents=tuple(agents),
            objects=tuple(
                sorted(
                    objects.values(),
                    key=lambda obj: obj.object_id,
                )
            ),
            recipe_completed=state.recipe_completed,
            bad_delivery_attempts=state.bad_delivery_attempts,
            t=0,
        )
        if self.projection_mode == "role_core":
            projected = self._role_core_projection(projected)
        return self._cache_safety_projection(state, projected)

    def _cache_safety_projection(
        self,
        state: CookingZooState,
        projected: CookingZooState,
    ) -> CookingZooState:
        if len(self._safety_projection_cache) >= self._state_cache_limit:
            self._safety_projection_cache.clear()
        self._safety_projection_cache[state] = projected
        return projected

    def _role_core_projection(self, state: CookingZooState) -> CookingZooState:
        objects_by_id = self._objects_by_id(state)
        object_id_mapping: dict[int, int] = {}
        updated_objects: dict[int, CookingZooObjectState] = {}
        agents: list[CookingZooAgentState] = []
        for agent_idx, agent in enumerate(state.agents):
            if agent.holding is None:
                agents.append(agent)
                continue
            held = objects_by_id.get(agent.holding)
            if held is None:
                agents.append(replace(agent, holding=None))
                continue
            role_key = self._held_role_core_key(held, agent_idx, state, objects_by_id)
            role_object_id = self._role_core_object_id(role_key)
            object_id_mapping[held.object_id] = role_object_id
            updated_objects[held.object_id] = replace(held, object_id=role_object_id)
            agents.append(replace(agent, holding=role_object_id))

        objects: list[CookingZooObjectState] = []
        for obj in state.objects:
            if self._role_core_drops_static_delta(obj):
                continue
            updated = updated_objects.get(obj.object_id, obj)
            if updated.content:
                updated = replace(
                    updated,
                    content=tuple(
                        object_id_mapping.get(content_id, content_id)
                        for content_id in updated.content
                    ),
                )
            objects.append(self._role_core_object_state(updated))

        return replace(
            state,
            agents=tuple(agents),
            objects=tuple(sorted(objects, key=lambda obj: obj.object_id)),
        )

    def _role_core_drops_static_delta(self, obj: CookingZooObjectState) -> bool:
        if obj.kind != "Counter" or obj.content:
            return False
        base_obj = self._projection_base_objects_by_id.get(obj.object_id)
        if base_obj is None or not base_obj.content:
            return False
        return self._attrs_dict(obj.attrs) == self._attrs_dict(base_obj.attrs)

    def _role_core_object_state(
        self,
        obj: CookingZooObjectState,
    ) -> CookingZooObjectState:
        attrs = self._role_core_attrs(obj.attrs)
        if attrs == obj.attrs:
            return obj
        return replace(obj, attrs=attrs)

    def _held_role_core_key(
        self,
        held: CookingZooObjectState,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> tuple[str, ...]:
        if held.object_id < 0:
            existing_key = self._role_core_keys_by_object_id.get(held.object_id)
            if existing_key is not None:
                return existing_key
        attr_key = tuple(
            f"{attr_name}={value}"
            for attr_name, value in self._role_core_attrs(held.attrs)
        )
        origin_key = f"origin={held.object_id}"
        if held.kind == "Plate":
            if self._plate_matches_recipe(held.object_id, agent_idx, state, objects):
                plate_role = "complete"
            elif held.content:
                plate_role = "incomplete"
            else:
                plate_role = "empty"
            content_key = tuple(
                self._role_core_content_token(objects.get(content_id))
                for content_id in held.content
            )
            return (
                "held",
                str(agent_idx),
                "plate",
                origin_key,
                plate_role,
                *content_key,
                *attr_key,
            )
        if self._is_food_kind(held.kind):
            if self._is_own_required_ingredient(held.kind, agent_idx):
                role = "own_ingredient"
            elif self._is_teammate_only_ingredient(held.kind, agent_idx):
                role = "teammate_only_ingredient"
            elif self._is_teammate_requested_item(held.kind, agent_idx, state):
                role = "teammate_requested_item"
            else:
                role = "irrelevant_food"
            return ("held", str(agent_idx), role, held.kind, origin_key, *attr_key)
        return ("held", str(agent_idx), "other", held.kind, origin_key, *attr_key)

    def _role_core_content_token(
        self,
        obj: CookingZooObjectState | None,
    ) -> str:
        if obj is None:
            return "missing"
        role_core_attrs = self._role_core_attrs(obj.attrs)
        key = (obj.kind, role_core_attrs)
        cached = self._role_core_content_token_cache.get(key)
        if cached is not None:
            return cached
        attrs = ",".join(
            f"{attr_name}={value}"
            for attr_name, value in role_core_attrs
        )
        token = f"{obj.kind}[{attrs}]"
        self._role_core_content_token_cache[key] = token
        return token

    def _attrs_dict(self, attrs: tuple[tuple[str, str], ...]) -> dict[str, str]:
        cached = self._attrs_dict_cache.get(attrs)
        if cached is None:
            cached = dict(attrs)
            self._attrs_dict_cache[attrs] = cached
        return cached

    def _role_core_attrs(
        self,
        attrs: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        cached = self._role_core_attrs_cache.get(attrs)
        if cached is None:
            cached = tuple(
                (attr_name, value)
                for attr_name, value in attrs
                if attr_name in self._role_core_attr_names
            )
            self._role_core_attrs_cache[attrs] = cached
        return cached

    def _role_core_object_id(self, key: tuple[str, ...]) -> int:
        existing = self._role_core_object_ids.get(key)
        if existing is not None:
            return existing
        digest = hashlib.blake2b(
            "\x1f".join(key).encode("utf-8"),
            digest_size=8,
        ).digest()
        object_id = -1 - int.from_bytes(digest, byteorder="big", signed=False)
        while self._role_core_keys_by_object_id.get(object_id) not in (None, key):
            object_id -= 1
        self._role_core_object_ids[key] = object_id
        self._role_core_keys_by_object_id[object_id] = key
        return object_id

    def _role_core_attribute_names(self) -> frozenset[str]:
        attr_names = {
            "walkable",
            "status",
            "toggle",
            "chop_state",
            "blend_state",
            "toast_state",
            "microwave_state",
            "boil_state",
        }
        for recipe in self._recipes:
            for node in recipe.node_list:
                attr_names.update(str(attr_name) for attr_name, _ in node.conditions)
        return frozenset(attr_names)

    def _station_state_changed(
        self,
        obj: CookingZooObjectState,
        base_obj: CookingZooObjectState | None,
    ) -> bool:
        if self._is_dynamic_kind(obj.kind) or obj.kind not in _STATION_KINDS:
            return False
        if base_obj is None:
            return True
        if obj.content != base_obj.content:
            return True
        attrs = self._attrs_dict(obj.attrs)
        base_attrs = self._attrs_dict(base_obj.attrs)
        for attr_name in ("status", "toggle"):
            if attrs.get(attr_name) != base_attrs.get(attr_name):
                return True
        return False

    def _is_projection_relevant_dynamic_kind(self, kind: str) -> bool:
        return kind == "Plate" or self._is_food_kind(kind)

    def _has_projection_relevant_content(self, content: tuple[int, ...]) -> bool:
        return any(
            content_id in self._projection_relevant_dynamic_ids
            for content_id in content
        )

    def local_actions(
        self,
        agent_id: str,
        state: CookingZooState,
    ) -> tuple[LocalAction, ...]:
        agent_idx = self._agent_index[agent_id]
        if agent_idx < len(state.agents) and not state.agents[agent_idx].active:
            return (_NOOP,)
        base_actions = self._movement_or_noop_actions
        if (
            agent_idx < len(state.agents)
            and state.agents[agent_idx].location
            in self._projection_fallback_locations
        ):
            return base_actions
        objects = self._objects_by_id(state)
        interaction_actions = self._enabled_interaction_actions(
            agent_idx,
            state,
            objects,
        )
        return tuple(
            action
            for action in self._actions
            if action in base_actions or action in interaction_actions
        )

    def _enabled_interaction_actions(
        self,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> frozenset[int]:
        if agent_idx >= len(state.agents):
            return frozenset()
        agent = state.agents[agent_idx]
        if not agent.active:
            return frozenset()
        target = self._target_location(agent.location, agent.orientation)
        if any(
            other_idx != agent_idx and other.location == target
            for other_idx, other in enumerate(state.agents)
        ):
            return frozenset()
        static = self._static_object_at(target, objects)
        if static is None:
            return frozenset()

        enabled: set[int] = set()
        dynamic_objects = self._dynamic_objects_at(target, objects)
        held = objects.get(agent.holding) if agent.holding is not None else None
        if self._primary_interaction_can_change_state(
            agent_idx,
            state,
            objects,
            static,
            dynamic_objects,
            held,
        ):
            enabled.add(_INTERACT_PRIMARY)
        if self._pick_up_special_can_change_state(held, dynamic_objects, objects):
            enabled.add(_INTERACT_PICK_UP_SPECIAL)
        if self._execute_action_can_change_state(static, objects):
            enabled.add(_EXECUTE_ACTION)
        return frozenset(enabled)

    def _primary_interaction_can_change_state(
        self,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
        static: CookingZooObjectState,
        dynamic_objects: list[CookingZooObjectState],
        held: CookingZooObjectState | None,
    ) -> bool:
        if held is None:
            return (
                self._pick_releasable_object(static, dynamic_objects, objects)
                is not None
            )
        if static.kind == "Deliversquare":
            return not static.content
        plate = next((obj for obj in dynamic_objects if obj.kind == "Plate"), None)
        if plate is not None and self._can_add_to_plate(
            plate.object_id,
            held.object_id,
            objects,
        ):
            return True
        if held.kind == "Plate" and any(
            self._is_food_kind(obj.kind) and self._food_done(obj)
            for obj in dynamic_objects
        ):
            return True
        if static.kind == "Cutboard" and self._is_fresh_chop_food(held):
            return True
        if static.kind == "Blender" and self._is_fresh_blender_food(held):
            return True
        if static.kind == "Counter" and not static.content:
            return True
        return False

    def _pick_up_special_can_change_state(
        self,
        held: CookingZooObjectState | None,
        dynamic_objects: list[CookingZooObjectState],
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        if held is not None:
            return False
        content_objects = [obj for obj in dynamic_objects if obj.content]
        if len(content_objects) != 1:
            return False
        return any(
            content_id in objects
            for content_id in content_objects[0].content
        )

    def _execute_action_can_change_state(
        self,
        static: CookingZooObjectState,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        if self._attrs_dict(static.attrs).get("status") != "READY":
            return False
        if static.kind == "Blender":
            return True
        if static.kind != "Cutboard":
            return False
        return any(
            (content := objects.get(content_id)) is not None
            and self._is_fresh_chop_food(content)
            for content_id in static.content
        )

    def label(self, state: CookingZooState) -> Label:
        labels: set[str] = set()
        objects = self._objects_by_id(state)

        for agent_idx, agent in enumerate(state.agents):
            agent_id = self.agent_ids[agent_idx]
            x, y = agent.location
            labels.add(f"{agent_id}_at_{x}_{y}")
            at_delivery_access = agent.location in self._delivery_access_cells
            at_cutboard_access = agent.location in self._cutboard_access_cells
            target_static = self._static_object_at(
                self._target_location(agent.location, agent.orientation),
                objects,
            )
            facing_delivery = (
                target_static is not None and target_static.kind == "Deliversquare"
            )
            facing_cutboard = (
                target_static is not None and target_static.kind == "Cutboard"
            )
            if at_delivery_access:
                labels.add(f"{agent_id}_at_delivery_access")
            if facing_delivery:
                labels.add(f"{agent_id}_facing_delivery")
            if at_cutboard_access:
                labels.add(f"{agent_id}_at_cutboard_access")
            if facing_cutboard:
                labels.add(f"{agent_id}_facing_cutboard")
            bad_delivery_attempt = agent_idx in state.bad_delivery_attempts
            held = objects.get(agent.holding) if agent.holding is not None else None
            blocking_delivery = (
                at_delivery_access
                and facing_delivery
                and held is not None
                and not self._plate_matches_recipe(
                    held.object_id,
                    agent_idx,
                    state,
                    objects,
                )
            )
            blocking_cutboard = (
                at_cutboard_access
                and facing_cutboard
                and held is not None
                and not self._is_fresh_chop_food(held)
            )
            if bad_delivery_attempt:
                labels.add(f"{agent_id}_bad_delivery_attempt")
            if blocking_delivery:
                labels.add(f"{agent_id}_blocking_delivery_access")
            if blocking_cutboard:
                labels.add(f"{agent_id}_blocking_cutboard_access")

            teammate_only = (
                held is not None
                and self._is_teammate_only_ingredient(held.kind, agent_idx)
            )
            if self._is_recipe_role_ok_held(held, agent_idx):
                labels.add(f"{agent_id}_recipe_role_ok")
            delivery_station_ok = (
                not at_delivery_access
                or not facing_delivery
                or (
                    held is not None
                    and held.kind == "Plate"
                    and self._plate_matches_recipe(
                        held.object_id,
                        agent_idx,
                        state,
                        objects,
                    )
                )
            )
            cutboard_station_ok = (
                not at_cutboard_access
                or not facing_cutboard
                or (held is not None and self._is_fresh_chop_food(held))
            )
            if not bad_delivery_attempt and delivery_station_ok:
                labels.add(f"{agent_id}_delivery_station_ok")
            if cutboard_station_ok:
                labels.add(f"{agent_id}_cutboard_station_ok")
            if delivery_station_ok and cutboard_station_ok:
                labels.add(f"{agent_id}_station_clearance_ok")
            if held is None:
                continue
            kind_label = self._label_token(held.kind)
            labels.add(f"{agent_id}_holding_{kind_label}")
            if self._is_own_required_ingredient(held.kind, agent_idx):
                labels.add(f"{agent_id}_holding_own_required_ingredient")
            if self._is_teammate_requested_item(held.kind, agent_idx, state):
                labels.add(f"{agent_id}_holding_teammate_requested_item")
            if facing_delivery and self._plate_matches_recipe(
                held.object_id,
                agent_idx,
                state,
                objects,
            ):
                labels.add(f"{agent_id}_ready_to_deliver")
            if facing_cutboard and self._is_fresh_chop_food(held):
                labels.add(f"{agent_id}_ready_to_chop")
            if held.kind == "Plate":
                labels.add(f"{agent_id}_holding_plate")
                if self._plate_matches_recipe(held.object_id, agent_idx, state, objects):
                    labels.add(f"{agent_id}_holding_complete_dish")
                else:
                    labels.add(f"{agent_id}_holding_incomplete_dish")
            elif teammate_only:
                labels.add(f"{agent_id}_holding_teammate_only_ingredient")

        for recipe_idx, completed in enumerate(state.recipe_completed):
            if completed:
                labels.add(f"recipe_{recipe_idx}_completed")
        if state.recipe_completed and all(state.recipe_completed):
            labels.add("all_recipes_completed")
        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = {"all_recipes_completed"}
        width = int(getattr(self.base_env, "width", getattr(self.world, "width", 0)) or 0)
        height = int(getattr(self.base_env, "height", getattr(self.world, "height", 0)) or 0)
        if width <= 0 or height <= 0:
            state = self.abstract_state(self.env)
            locations = [agent.location for agent in state.agents]
            locations.extend(obj.location for obj in state.objects)
            width = max((x for x, _ in locations), default=0) + 1
            height = max((y for _, y in locations), default=0) + 1
        object_kinds = {
            self._label_token(kind)
            for object_list in self.world.world_objects.values()
            for obj in object_list
            for kind in (getattr(obj, "kind", None) or type(obj).__name__,)
            if self._is_dynamic_kind(kind)
        }
        for agent_id in self.agent_ids:
            for y in range(height):
                for x in range(width):
                    labels.add(f"{agent_id}_at_{x}_{y}")
            labels.add(f"{agent_id}_at_delivery_access")
            labels.add(f"{agent_id}_facing_delivery")
            labels.add(f"{agent_id}_at_cutboard_access")
            labels.add(f"{agent_id}_facing_cutboard")
            labels.add(f"{agent_id}_bad_delivery_attempt")
            labels.add(f"{agent_id}_blocking_delivery_access")
            labels.add(f"{agent_id}_blocking_cutboard_access")
            labels.add(f"{agent_id}_holding_plate")
            labels.add(f"{agent_id}_holding_complete_dish")
            labels.add(f"{agent_id}_holding_incomplete_dish")
            labels.add(f"{agent_id}_holding_own_required_ingredient")
            labels.add(f"{agent_id}_holding_teammate_requested_item")
            labels.add(f"{agent_id}_holding_teammate_only_ingredient")
            labels.add(f"{agent_id}_ready_to_deliver")
            labels.add(f"{agent_id}_ready_to_chop")
            for family in COOKING_ZOO_PROTOCOL_LABEL_FAMILIES:
                labels.add(f"{agent_id}_{family}")
            for kind_label in object_kinds:
                labels.add(f"{agent_id}_holding_{kind_label}")
        for recipe_idx in range(len(self._recipes)):
            labels.add(f"recipe_{recipe_idx}_completed")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return cooking_zoo_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return cooking_zoo_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def local_shield_formulas_by_agent(
        self,
        *,
        formula_name: str | None,
        global_formula: str,
    ) -> dict[str, str] | None:
        if formula_name != "kitchen_etiquette":
            return None
        return {
            agent_id: self._kitchen_etiquette_obligation(
                agent_id,
                include_station_protocol=False,
            )
            for agent_id in self.agent_ids
        }

    def contract_seed_formulas_by_agent(
        self,
        *,
        global_formula: str,
    ) -> dict[str, str]:
        suffixes = (
            "bad_delivery_attempt",
            "holding_teammate_only_ingredient",
            "blocking_delivery_access",
            "blocking_cutboard_access",
        )
        seed_formulas: dict[str, str] = {}
        for agent_id in self.agent_ids:
            mentioned = tuple(
                suffix
                for suffix in suffixes
                if f"{agent_id}_{suffix}" in global_formula
                or f"{{agent}}_{suffix}" in global_formula
            )
            if not mentioned:
                continue
            if len(mentioned) == len(suffixes):
                seed_formulas[agent_id] = self._kitchen_etiquette_obligation(
                    agent_id,
                    include_station_protocol=True,
                )
                continue
            conjuncts = tuple(f"!({agent_id}_{suffix})" for suffix in mentioned)
            seed_formulas[agent_id] = "G((" + ") & (".join(conjuncts) + "))"
        return seed_formulas

    @staticmethod
    def _kitchen_etiquette_obligation(
        agent_id: str,
        *,
        include_station_protocol: bool,
    ) -> str:
        conjuncts = [
            f"!({agent_id}_bad_delivery_attempt)",
            f"!({agent_id}_holding_teammate_only_ingredient)",
            f"!({agent_id}_blocking_delivery_access)",
            f"!({agent_id}_blocking_cutboard_access)",
            f"{agent_id}_recipe_role_ok",
        ]
        if include_station_protocol:
            conjuncts.extend(
                (
                    f"{agent_id}_delivery_station_ok",
                    f"{agent_id}_cutboard_station_ok",
                    f"{agent_id}_station_clearance_ok",
                )
            )
        return "G((" + ") & (".join(conjuncts) + "))"

    def joint_actions(
        self,
        state: CookingZooState,
    ) -> tuple[tuple[LocalAction, ...], ...]:
        return tuple(product(*(self.local_actions(agent_id, state) for agent_id in self.agent_ids)))

    def successors_for_joint_action(
        self,
        state: CookingZooState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[CookingZooState]:
        return frozenset(
            (
                self._successor_for_joint_action(
                    state,
                    tuple(int(action) for action in joint_action),
                ),
            )
        )

    def transition_outcomes_for_joint_action(
        self,
        state: CookingZooState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[CookingZooState], ...]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got {len(normalized_action)}."
            )
        next_state = self._successor_for_joint_action(state, normalized_action)
        rewards = self._rewards_for_transition(state, next_state)
        recipe_dones = (
            all(next_state.recipe_completed)
            if bool(getattr(self.base_env, "end_condition_all_dishes", False))
            else any(next_state.recipe_completed)
        )
        truncated = bool(self._max_steps and next_state.t >= self._max_steps)
        return (
            AbstractTransitionOutcome(
                next_state=next_state,
                probability=1.0,
                rewards={
                    agent_id: float(rewards[index])
                    for index, agent_id in enumerate(self.agent_ids)
                },
                terminations={agent_id: bool(recipe_dones) for agent_id in self.agent_ids},
                truncations={agent_id: truncated for agent_id in self.agent_ids},
                label=self.label(next_state),
            ),
        )

    def successors_for_local_action(
        self,
        state: CookingZooState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[CookingZooState]:
        cache_key = (state, agent_id, int(action))
        cached = self._successor_cache.get(cache_key)
        if cached is not None:
            return cached

        agent_idx = self._agent_index[agent_id]
        successors: set[CookingZooState] = set()
        other_agent_ids = [
            other_agent_id
            for other_agent_id in self.agent_ids
            if other_agent_id != agent_id
        ]
        other_action_sets = tuple(
            self._teammate_context_actions(other_agent_id, state)
            for other_agent_id in other_agent_ids
        )
        for other_actions in product(*other_action_sets):
            joint_action_by_agent = dict(zip(other_agent_ids, other_actions, strict=True))
            joint_action_by_agent[agent_id] = int(action)
            joint_action = [
                int(joint_action_by_agent[local_agent_id])
                for local_agent_id in self.agent_ids
            ]
            successors.update(self.successors_for_joint_action(state, tuple(joint_action)))
        result = frozenset(successors)
        self._successor_cache[cache_key] = result
        return result

    def _teammate_context_actions(
        self,
        agent_id: str,
        state: CookingZooState,
    ) -> tuple[LocalAction, ...]:
        if self.teammate_context_mode == "enabled":
            return self.local_actions(agent_id, state)
        agent_idx = self._agent_index[agent_id]
        if agent_idx < len(state.agents) and not state.agents[agent_idx].active:
            return (_NOOP,)
        return self._movement_or_noop_actions

    def _successor_for_joint_action(
        self,
        state: CookingZooState,
        joint_action: tuple[int, ...],
    ) -> CookingZooState:
        agents = list(state.agents)
        objects = dict(self._objects_by_id(state))
        bad_delivery_attempts: set[int] = set()

        oriented_agents: list[CookingZooAgentState] = []
        for agent, action in zip(agents, joint_action, strict=True):
            if int(action) in _WALK_ACTIONS:
                oriented_agents.append(
                    CookingZooAgentState(
                        location=agent.location,
                        orientation=int(action),
                        holding=agent.holding,
                        active=agent.active,
                    )
                )
            else:
                oriented_agents.append(agent)
        agents = oriented_agents

        effective_actions = self._collision_checked_actions(agents, joint_action, objects)
        for agent_idx, action in enumerate(effective_actions):
            agent = agents[agent_idx]
            if not agent.active:
                continue
            if int(action) in _WALK_ACTIONS:
                agents[agent_idx] = self._move_agent(agent, int(action), objects)
                continue
            if int(action) == _INTERACT_PRIMARY:
                bad_delivery = self._resolve_primary_interaction(
                    agents,
                    agent_idx,
                    objects,
                )
                if bad_delivery:
                    bad_delivery_attempts.add(agent_idx)
            elif int(action) == _INTERACT_PICK_UP_SPECIAL:
                self._resolve_pick_up_special(agents, agent_idx, objects)
            elif int(action) == _EXECUTE_ACTION:
                self._resolve_execute_action(agents, agent_idx, objects)

        self._progress_world(objects)
        next_t = int(state.t) + 1
        if self._max_steps and next_t >= self._max_steps:
            agents = [
                CookingZooAgentState(
                    location=agent.location,
                    orientation=agent.orientation,
                    holding=agent.holding,
                    active=False,
                )
                for agent in agents
            ]

        next_agents = tuple(agents)
        next_objects = tuple(sorted(objects.values(), key=lambda obj: obj.object_id))
        next_bad_delivery_attempts = frozenset(bad_delivery_attempts)
        partial_next_state = CookingZooState(
            agents=next_agents,
            objects=next_objects,
            recipe_completed=state.recipe_completed,
            bad_delivery_attempts=next_bad_delivery_attempts,
            t=next_t,
        )
        recipe_completed = self._recipe_completion_from_state(partial_next_state, objects)
        return CookingZooState(
            agents=next_agents,
            objects=next_objects,
            recipe_completed=recipe_completed,
            bad_delivery_attempts=next_bad_delivery_attempts,
            t=next_t,
        )

    def _rewards_for_transition(
        self,
        state: CookingZooState,
        next_state: CookingZooState,
    ) -> tuple[float, ...]:
        recipe_reward = float(self._reward_scheme.get("recipe_reward", 20.0))
        recipe_penalty = float(self._reward_scheme.get("recipe_penalty", -40.0))
        max_time_penalty = float(self._reward_scheme.get("max_time_penalty", -5.0))
        step_penalty = max_time_penalty / float(self._max_steps) if self._max_steps else 0.0

        rewards: list[float] = []
        for before, after in zip(
            state.recipe_completed,
            next_state.recipe_completed,
            strict=True,
        ):
            reward = step_penalty
            if after and not before:
                reward += recipe_reward
            if before and not after:
                reward += recipe_penalty
            rewards.append(float(reward))
        return tuple(rewards)

    def _progress_world(self, objects: dict[int, CookingZooObjectState]) -> None:
        for obj in tuple(objects.values()):
            if obj.kind == "Blender" and self._attrs_dict(obj.attrs).get("toggle") == "True":
                self._progress_blender(obj, objects)

        for obj in tuple(objects.values()):
            if not obj.content:
                continue
            for content_id in obj.content:
                content = objects.get(content_id)
                if content is not None:
                    objects[content_id] = self._replace_object(
                        content,
                        location=obj.location,
                        free=False,
                    )
            top = objects.get(obj.content[-1])
            if top is not None:
                objects[obj.content[-1]] = self._replace_object(
                    top,
                    location=obj.location,
                    free=True,
                )

    def _progress_blender(
        self,
        blender: CookingZooObjectState,
        objects: dict[int, CookingZooObjectState],
    ) -> None:
        if not blender.content:
            return
        all_mashed = True
        for content_id in blender.content:
            content = objects.get(content_id)
            if content is None or "blend_state" not in self._attrs_dict(content.attrs):
                continue
            attrs = dict(content.attrs)
            if attrs.get("blend_state") == "MASHED":
                continue
            progress = int(attrs.get("current_progress", "1")) - 1
            max_progress = int(attrs.get("max_progress", "0"))
            attrs["current_progress"] = str(progress)
            attrs["blend_state"] = "IN_PROGRESS" if progress > max_progress else "MASHED"
            objects[content_id] = self._replace_object(
                content,
                attrs=tuple(sorted(attrs.items())),
            )
            all_mashed = all_mashed and attrs["blend_state"] == "MASHED"
        if all_mashed:
            objects[blender.object_id] = self._replace_attr(
                self._replace_attr(blender, "toggle", "False"),
                "status",
                "NOT_USABLE",
            )
            for content_id in blender.content:
                content = objects.get(content_id)
                if content is None:
                    continue
                attrs = dict(content.attrs)
                if "min_progress" in attrs:
                    attrs["current_progress"] = attrs["min_progress"]
                    objects[content_id] = self._replace_object(
                        content,
                        attrs=tuple(sorted(attrs.items())),
                    )

    def _object_state(self, obj: object) -> CookingZooObjectState:
        attrs: list[tuple[str, str]] = []
        for attr_name in (
            "chop_state",
            "blend_state",
            "toast_state",
            "microwave_state",
            "boil_state",
            "status",
            "current_progress",
            "max_progress",
            "min_progress",
        ):
            if not hasattr(obj, attr_name):
                continue
            attrs.append((attr_name, self._stable_value(getattr(obj, attr_name))))
        for attr_name in ("toggle", "walkable"):
            if hasattr(obj, attr_name):
                attrs.append((attr_name, self._stable_value(getattr(obj, attr_name))))
        content = tuple(
            int(content_obj.unique_id)
            for content_obj in getattr(obj, "content", ())
            if hasattr(content_obj, "unique_id")
        )
        return CookingZooObjectState(
            object_id=int(obj.unique_id),
            kind=type(obj).__name__,
            location=tuple(obj.location),
            attrs=tuple(sorted(attrs)),
            content=content,
            free=bool(getattr(obj, "free", True)),
        )

    def _without_agent_contents(
        self,
        obj: CookingZooObjectState,
        agent_object_ids: frozenset[int],
    ) -> CookingZooObjectState:
        if not obj.content or not agent_object_ids:
            return obj
        filtered_content = tuple(
            content_id
            for content_id in obj.content
            if content_id not in agent_object_ids
        )
        if filtered_content == obj.content:
            return obj
        return replace(obj, content=filtered_content)

    def _agent_has_bad_delivery_interaction(
        self,
        agent: object,
        agent_idx: int,
        object_states: tuple[CookingZooObjectState, ...],
        recipe_completed: tuple[bool, ...],
    ) -> bool:
        if not any(type(obj).__name__ == "Deliversquare" for obj in getattr(agent, "interacts_with", ())):
            return False
        if agent_idx < len(recipe_completed) and recipe_completed[agent_idx]:
            return False
        objects = {obj.object_id: obj for obj in object_states}
        for delivery in object_states:
            if delivery.kind != "Deliversquare":
                continue
            for content_id in delivery.content:
                content = objects.get(content_id)
                if content is not None and content.kind == "Plate":
                    if self._plate_matches_recipe(
                        content.object_id,
                        agent_idx,
                        CookingZooState((), object_states, recipe_completed),
                        objects,
                    ):
                        return False
        return True

    def _collision_checked_actions(
        self,
        agents: list[CookingZooAgentState],
        joint_action: tuple[int, ...],
        objects: dict[int, CookingZooObjectState],
    ) -> list[int]:
        target_locations: list[tuple[int, int]] = []
        walkable_targets: list[bool] = []
        cleaned_actions: list[int] = []
        for agent, action in zip(agents, joint_action, strict=True):
            action = int(action)
            if action not in _WALK_ACTIONS:
                cleaned_actions.append(action)
                target_locations.append(agent.location)
                walkable_targets.append(False)
                continue
            target = self._target_location(agent.location, action)
            walkable = self._is_walkable(target, objects)
            cleaned_action = action if walkable else _NOOP
            cleaned_actions.append(cleaned_action)
            target_locations.append(target if walkable else agent.location)
            walkable_targets.append(walkable)

        collision_checked: list[int] = []
        for idx, (action, target, walkable) in enumerate(
            zip(cleaned_actions, target_locations, walkable_targets, strict=True)
        ):
            duplicate_target = target in target_locations[:idx] + target_locations[idx + 1 :]
            collision_checked.append(_NOOP if duplicate_target and walkable else action)
        return collision_checked

    def _move_agent(
        self,
        agent: CookingZooAgentState,
        action: int,
        objects: dict[int, CookingZooObjectState],
    ) -> CookingZooAgentState:
        target = self._target_location(agent.location, action)
        if not self._is_walkable(target, objects):
            return agent
        if agent.holding is not None and agent.holding in objects:
            objects[agent.holding] = self._replace_object(
                objects[agent.holding],
                location=target,
            )
        return CookingZooAgentState(
            location=target,
            orientation=action,
            holding=agent.holding,
            active=agent.active,
        )

    def _resolve_primary_interaction(
        self,
        agents: list[CookingZooAgentState],
        agent_idx: int,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        agent = agents[agent_idx]
        target = self._target_location(agent.location, agent.orientation)
        if any(
            other_idx != agent_idx and other.location == target
            for other_idx, other in enumerate(agents)
        ):
            return False
        static = self._static_object_at(target, objects)
        if static is None:
            return False
        dynamic_objects = self._dynamic_objects_at(target, objects)

        if agent.holding is None:
            grabbed = self._pick_releasable_object(static, dynamic_objects, objects)
            if grabbed is None:
                return False
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=grabbed.object_id,
                active=agent.active,
            )
            objects[grabbed.object_id] = self._replace_object(
                grabbed,
                location=agent.location,
            )
            objects[static.object_id] = self._after_content_removed(
                self._remove_content(static, grabbed.object_id)
            )
            return False

        held = objects.get(agent.holding)
        if held is None:
            return False
        if static.kind == "Deliversquare":
            if static.content:
                return False
            held = self._materialize_released_object(held, objects)
            bad_delivery = not self._plate_matches_recipe(
                held.object_id,
                agent_idx,
                CookingZooState(tuple(agents), tuple(objects.values()), ()),
                objects,
            )
            objects[held.object_id] = self._replace_object(held, location=target)
            objects[static.object_id] = self._append_content(static, held.object_id)
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=None,
                active=agent.active,
            )
            return bad_delivery

        plate = next((obj for obj in dynamic_objects if obj.kind == "Plate"), None)
        if plate is not None and self._can_add_to_plate(plate.object_id, held.object_id, objects):
            held = self._materialize_released_object(held, objects)
            objects[held.object_id] = self._replace_object(held, location=plate.location)
            objects[plate.object_id] = self._append_content(plate, held.object_id)
            objects[static.object_id] = self._after_content_removed(
                self._remove_content(static, held.object_id)
            )
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=None,
                active=agent.active,
            )
            return False

        if held.kind == "Plate":
            done_food = next(
                (
                    obj
                    for obj in dynamic_objects
                    if self._is_food_kind(obj.kind) and self._food_done(obj)
                ),
                None,
            )
            if done_food is not None:
                objects[done_food.object_id] = self._replace_object(
                    done_food,
                    location=agent.location,
                )
                objects[held.object_id] = self._append_content(held, done_food.object_id)
                objects[static.object_id] = self._after_content_removed(
                    self._remove_content(
                        static,
                        done_food.object_id,
                    )
                )
                return False

        if static.kind == "Cutboard" and self._is_fresh_chop_food(held):
            held = self._materialize_released_object(held, objects)
            objects[held.object_id] = self._replace_object(held, location=target)
            objects[static.object_id] = self._replace_attr(
                self._append_content(static, held.object_id),
                "status",
                "READY",
            )
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=None,
                active=agent.active,
            )
            return False

        if static.kind == "Blender" and self._is_fresh_blender_food(held):
            held = self._materialize_released_object(held, objects)
            objects[held.object_id] = self._replace_object(held, location=target)
            objects[static.object_id] = self._replace_attr(
                self._append_content(static, held.object_id),
                "status",
                "READY",
            )
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=None,
                active=agent.active,
            )
            return False

        if (
            static.kind == "Counter"
            and not static.content
        ):
            held = self._materialize_released_object(held, objects)
            objects[held.object_id] = self._replace_object(held, location=target)
            objects[static.object_id] = self._append_content(static, held.object_id)
            agents[agent_idx] = CookingZooAgentState(
                location=agent.location,
                orientation=agent.orientation,
                holding=None,
                active=agent.active,
            )
        return False

    def _resolve_pick_up_special(
        self,
        agents: list[CookingZooAgentState],
        agent_idx: int,
        objects: dict[int, CookingZooObjectState],
    ) -> None:
        agent = agents[agent_idx]
        if agent.holding is not None:
            return
        target = self._target_location(agent.location, agent.orientation)
        content_objects = [
            obj
            for obj in self._dynamic_objects_at(target, objects)
            if obj.content
        ]
        if len(content_objects) != 1:
            return
        source = content_objects[0]
        grabbed_id = source.content[-1]
        grabbed = objects.get(grabbed_id)
        if grabbed is None:
            return
        objects[source.object_id] = self._after_content_removed(
            self._remove_content(source, grabbed_id)
        )
        objects[grabbed_id] = self._replace_object(grabbed, location=agent.location)
        agents[agent_idx] = CookingZooAgentState(
            location=agent.location,
            orientation=agent.orientation,
            holding=grabbed_id,
            active=agent.active,
        )

    def _resolve_execute_action(
        self,
        agents: list[CookingZooAgentState],
        agent_idx: int,
        objects: dict[int, CookingZooObjectState],
    ) -> None:
        agent = agents[agent_idx]
        target = self._target_location(agent.location, agent.orientation)
        static = self._static_object_at(target, objects)
        if static is None:
            return
        if self._attrs_dict(static.attrs).get("status") != "READY":
            return
        if static.kind == "Blender":
            objects[static.object_id] = self._replace_attr(static, "toggle", "True")
            return
        if static.kind != "Cutboard":
            return
        for content_id in static.content:
            content = objects.get(content_id)
            if content is None or not self._is_fresh_chop_food(content):
                continue
            objects[content_id] = self._replace_attr(content, "chop_state", "CHOPPED")
            objects[static.object_id] = self._replace_attr(static, "status", "NOT_USABLE")
            break

    def _recipe_completion_from_state(
        self,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> tuple[bool, ...]:
        completed: list[bool] = []
        for recipe_idx in range(len(self._recipes)):
            matched = False
            for delivery in objects.values():
                if delivery.kind != "Deliversquare":
                    continue
                for content_id in delivery.content:
                    content = objects.get(content_id)
                    if content is None or content.kind != "Plate":
                        continue
                    if self._plate_matches_recipe(content_id, recipe_idx, state, objects):
                        matched = True
                        break
                if matched:
                    break
            completed.append(matched)
        return tuple(completed)

    def _plate_matches_recipe(
        self,
        plate_id: int,
        recipe_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState] | None = None,
    ) -> bool:
        if recipe_idx >= len(self._recipes):
            return False
        objects = objects or self._objects_by_id(state)
        plate = objects.get(plate_id)
        if plate is None or plate.kind != "Plate":
            return False
        plate_node = self._plate_node(self._recipes[recipe_idx])
        if plate_node is None:
            return False
        return self._content_matches_nodes(plate.content, tuple(plate_node.contains), objects)

    def _content_matches_nodes(
        self,
        content_ids: tuple[int, ...],
        nodes: tuple[RecipeNode, ...],
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        remaining = list(content_ids)
        for node in nodes:
            match_index = next(
                (
                    idx
                    for idx, object_id in enumerate(remaining)
                    if self._object_matches_node(objects.get(object_id), node, objects)
                ),
                None,
            )
            if match_index is None:
                return False
            remaining.pop(match_index)
        return True

    def _object_matches_node(
        self,
        obj: CookingZooObjectState | None,
        node: RecipeNode,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        if obj is None or obj.kind != node.name:
            return False
        attrs = self._attrs_dict(obj.attrs)
        for attr_name, expected in node.conditions:
            if attrs.get(attr_name) != self._stable_value(expected):
                return False
        if node.contains:
            return self._content_matches_nodes(obj.content, tuple(node.contains), objects)
        return True

    def _plate_node(self, recipe: Recipe) -> RecipeNode | None:
        for node in recipe.node_list:
            if node.name == "Plate":
                return node
        return None

    def _recipe_food_names(self, recipe: Recipe) -> frozenset[str]:
        return frozenset(
            node.name
            for node in recipe.node_list
            if node.name not in _RECIPE_CONTAINER_KINDS and not node.contains
        )

    def _is_teammate_only_ingredient(self, kind: str, agent_idx: int) -> bool:
        if not self._is_food_kind(kind):
            return False
        local_required = self._recipe_food_requirements[agent_idx]
        teammate_required = set().union(
            *(
                requirements
                for idx, requirements in enumerate(self._recipe_food_requirements)
                if idx != agent_idx
            )
        )
        return kind in teammate_required and kind not in local_required

    def _is_recipe_role_ok_held(
        self,
        held: CookingZooObjectState | None,
        agent_idx: int,
    ) -> bool:
        if held is None:
            return True
        if held.kind == "Plate":
            return True
        if self._is_food_kind(held.kind):
            return self._is_own_required_ingredient(held.kind, agent_idx)
        return False

    def _is_delivery_station_ok(
        self,
        agent: CookingZooAgentState,
        held: CookingZooObjectState | None,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        if agent.location not in self._delivery_access_cells:
            return True
        if not self._is_facing_station_kind(agent, "Deliversquare", objects):
            return True
        if held is None or held.kind != "Plate":
            return False
        return self._plate_matches_recipe(held.object_id, agent_idx, state, objects)

    def _is_cutboard_station_ok(
        self,
        agent: CookingZooAgentState,
        held: CookingZooObjectState | None,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        if agent.location not in self._cutboard_access_cells:
            return True
        if not self._is_facing_station_kind(agent, "Cutboard", objects):
            return True
        if held is None:
            return False
        return self._is_fresh_chop_food(held)

    def _is_own_required_ingredient(self, kind: str, agent_idx: int) -> bool:
        if not self._is_food_kind(kind):
            return False
        if agent_idx >= len(self._recipe_food_requirements):
            return False
        return kind in self._recipe_food_requirements[agent_idx]

    def _is_teammate_requested_item(
        self,
        kind: str,
        agent_idx: int,
        state: CookingZooState,
    ) -> bool:
        if not self._is_food_kind(kind):
            return False
        for recipe_idx, requirements in enumerate(self._recipe_food_requirements):
            if recipe_idx == agent_idx:
                continue
            if recipe_idx < len(state.recipe_completed) and state.recipe_completed[recipe_idx]:
                continue
            if kind in requirements:
                return True
        return False

    def _is_blocking_delivery_access(
        self,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        agent = state.agents[agent_idx]
        if agent.location not in self._delivery_access_cells:
            return False
        if not self._is_facing_station_kind(agent, "Deliversquare", objects):
            return False
        held = objects.get(agent.holding) if agent.holding is not None else None
        if held is None:
            return False
        return not self._plate_matches_recipe(held.object_id, agent_idx, state, objects)

    def _is_blocking_cutboard_access(
        self,
        agent_idx: int,
        state: CookingZooState,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        agent = state.agents[agent_idx]
        if agent.location not in self._cutboard_access_cells:
            return False
        if not self._is_facing_station_kind(agent, "Cutboard", objects):
            return False
        held = objects.get(agent.holding) if agent.holding is not None else None
        if held is None:
            return False
        return not self._is_fresh_chop_food(held)

    def _is_facing_station_kind(
        self,
        agent: CookingZooAgentState,
        station_kind: str,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        target = self._target_location(agent.location, agent.orientation)
        static = self._static_object_at(target, objects)
        return static is not None and static.kind == station_kind

    def _can_add_to_plate(
        self,
        plate_id: int,
        food_id: int,
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        plate = objects.get(plate_id)
        food = objects.get(food_id)
        return (
            plate is not None
            and plate.kind == "Plate"
            and food is not None
            and self._is_food_kind(food.kind)
            and self._food_done(food)
        )

    def _food_done(self, obj: CookingZooObjectState) -> bool:
        attrs = self._attrs_dict(obj.attrs)
        return attrs.get("chop_state") == "CHOPPED" or attrs.get("blend_state") == "MASHED"

    def _is_fresh_chop_food(self, obj: CookingZooObjectState) -> bool:
        return (
            obj.kind in _CHOP_FOOD_KINDS
            and self._attrs_dict(obj.attrs).get("chop_state") == "FRESH"
        )

    def _is_fresh_blender_food(self, obj: CookingZooObjectState) -> bool:
        return (
            obj.kind in _BLENDER_FOOD_KINDS
            and self._attrs_dict(obj.attrs).get("blend_state") == "FRESH"
        )

    def _is_food_kind(self, kind: str) -> bool:
        return kind in _FOOD_KINDS

    def _is_dynamic_kind(self, kind: str) -> bool:
        return kind in _DYNAMIC_KINDS

    def _dynamic_objects_at(
        self,
        location: tuple[int, int],
        objects: dict[int, CookingZooObjectState],
    ) -> list[CookingZooObjectState]:
        return [
            obj
            for obj in objects.values()
            if obj.location == location and self._is_dynamic_kind(obj.kind)
        ]

    def _static_object_at(
        self,
        location: tuple[int, int],
        objects: dict[int, CookingZooObjectState],
    ) -> CookingZooObjectState | None:
        for obj in objects.values():
            if obj.location != location:
                continue
            if not self._is_dynamic_kind(obj.kind):
                return obj
        return None

    def _pick_releasable_object(
        self,
        static: CookingZooObjectState,
        dynamic_objects: list[CookingZooObjectState],
        objects: dict[int, CookingZooObjectState],
    ) -> CookingZooObjectState | None:
        if static.kind == "Deliversquare":
            return None
        if static.kind == "Blender" and self._attrs_dict(static.attrs).get("toggle") == "True":
            return None
        if not dynamic_objects:
            return None
        for obj in dynamic_objects:
            if obj.free:
                return obj
        return dynamic_objects[-1]

    def _is_walkable(
        self,
        location: tuple[int, int],
        objects: dict[int, CookingZooObjectState],
    ) -> bool:
        static = self._static_object_at(location, objects)
        if static is None:
            return False
        return self._attrs_dict(static.attrs).get("walkable") == "True"

    def _station_access_cells(self, station_kind: str) -> frozenset[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for station in self.world.world_objects.get(station_kind, ()):
            for cell in self._adjacent_cells(tuple(station.location)):
                try:
                    if self.world.square_walkable(cell):
                        cells.add(cell)
                except Exception:
                    continue
        return frozenset(cells)

    def _station_projection_maps(
        self,
    ) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
        canonical_access: dict[tuple[str, int, int], tuple[int, int]] = {}
        canonical_entry: dict[tuple[str, int, int], tuple[int, int]] = {}
        raw: list[tuple[tuple[int, int], tuple[str, int, int], tuple[int, int] | None]] = []
        for station_kind in _STATION_KINDS:
            for station in self.world.world_objects.get(station_kind, ()):
                station_cell = tuple(station.location)
                for access_cell in self._adjacent_cells(station_cell):
                    try:
                        if not self.world.square_walkable(access_cell):
                            continue
                    except Exception:
                        continue
                    direction = (
                        station_cell[0] - access_cell[0],
                        station_cell[1] - access_cell[1],
                    )
                    key = (station_kind, direction[0], direction[1])
                    canonical_access.setdefault(key, access_cell)
                    entry_cell = (
                        access_cell[0] - direction[0],
                        access_cell[1] - direction[1],
                    )
                    try:
                        entry = entry_cell if self.world.square_walkable(entry_cell) else None
                    except Exception:
                        entry = None
                    if entry is not None:
                        canonical_entry.setdefault(key, entry)
                    raw.append((access_cell, key, entry))

        access_projection: dict[tuple[int, int], tuple[int, int]] = {}
        entry_projection: dict[tuple[int, int], tuple[int, int]] = {}
        for access_cell, key, entry_cell in raw:
            access_projection[access_cell] = canonical_access[key]
            if entry_cell is not None and key in canonical_entry:
                entry_projection[entry_cell] = canonical_entry[key]
        return access_projection, entry_projection

    def _station_relevant_cells(self) -> set[tuple[int, int]]:
        return set(self._station_access_projection.values()) | set(
            self._station_entry_projection.values()
        )

    def _content_access_cells(
        self,
        objects: tuple[CookingZooObjectState, ...],
    ) -> frozenset[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for obj in objects:
            if not self._has_projection_relevant_content(obj.content):
                continue
            for cell in self._adjacent_cells(obj.location):
                try:
                    if self.world.square_walkable(cell):
                        cells.add(cell)
                except Exception:
                    continue
        return frozenset(cells)

    def _project_location(
        self,
        location: tuple[int, int],
        fallback_location: tuple[int, int],
    ) -> tuple[int, int]:
        if location in self._station_access_projection:
            return self._station_access_projection[location]
        if location in self._station_entry_projection:
            return self._station_entry_projection[location]
        if location in self._projection_interesting_cells:
            return location
        try:
            if self.world.square_walkable(location):
                return fallback_location
        except Exception:
            pass
        return fallback_location

    def _project_orientation(
        self,
        location: tuple[int, int],
        orientation: int,
    ) -> int:
        if location in self._orientation_relevant_cells:
            return int(orientation)
        return _NOOP

    def _safe_projection_fallback_locations(
        self,
        initial_locations: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        access_cells = (
            set(self._delivery_access_cells)
            | set(self._cutboard_access_cells)
            | set(self._relevant_content_access_cells)
        )
        walkable: list[tuple[int, int]] = []
        width = int(getattr(self.base_env, "width", getattr(self.world, "width", 0)) or 0)
        height = int(getattr(self.base_env, "height", getattr(self.world, "height", 0)) or 0)
        for y in range(height):
            for x in range(width):
                cell = (x, y)
                if cell in access_cells:
                    continue
                try:
                    if self.world.square_walkable(cell):
                        walkable.append(cell)
                except Exception:
                    continue
        if not walkable:
            return initial_locations

        fallbacks: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        for initial in initial_locations:
            if initial in walkable and initial not in used:
                chosen = initial
            else:
                chosen = next(
                    (cell for cell in walkable if cell not in used),
                    walkable[0],
                )
            fallbacks.append(chosen)
            used.add(chosen)
        return tuple(fallbacks)

    def _target_location(
        self,
        location: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        x, y = location
        if action == int(ActionScheme1.WALK_LEFT):
            return (x - 1, y)
        if action == int(ActionScheme1.WALK_RIGHT):
            return (x + 1, y)
        if action == int(ActionScheme1.WALK_DOWN):
            return (x, y + 1)
        if action == int(ActionScheme1.WALK_UP):
            return (x, y - 1)
        return location

    def _adjacent_cells(self, location: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        x, y = location
        return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))

    def _objects_by_id(
        self,
        state: CookingZooState,
    ) -> dict[int, CookingZooObjectState]:
        cached = self._objects_by_id_cache.get(state)
        if cached is not None:
            return cached
        objects = dict(self._projection_base_objects_by_id)
        objects.update({obj.object_id: obj for obj in state.objects})
        explicit_content_owner = {
            content_id: obj.object_id
            for obj in state.objects
            for content_id in obj.content
        }
        if explicit_content_owner:
            for obj_id, obj in tuple(objects.items()):
                filtered_content = tuple(
                    content_id
                    for content_id in obj.content
                    if explicit_content_owner.get(content_id, obj_id) == obj_id
                )
                if filtered_content != obj.content:
                    objects[obj_id] = replace(obj, content=filtered_content)
        if self.projection_mode == "role_core":
            explicit_positive_dynamic_ids = {
                obj.object_id
                for obj in state.objects
                if obj.object_id in self._projection_dynamic_ids
            }
            held_origin_ids = {
                origin_id
                for agent in state.agents
                if agent.holding is not None
                and agent.holding < 0
                and (held := objects.get(agent.holding)) is not None
                and (origin_id := self._role_core_origin_id(held)) is not None
                and origin_id not in explicit_positive_dynamic_ids
            }
            if held_origin_ids:
                for holder_id, holder in tuple(objects.items()):
                    filtered_content = tuple(
                        content_id
                        for content_id in holder.content
                        if content_id not in held_origin_ids
                    )
                    if filtered_content != holder.content:
                        objects[holder_id] = replace(holder, content=filtered_content)
                for obj_id in held_origin_ids:
                    objects.pop(obj_id, None)
            referenced_dynamic_ids = {
                content_id
                for obj in objects.values()
                for content_id in obj.content
            }
            referenced_dynamic_ids.update(
                agent.holding
                for agent in state.agents
                if agent.holding is not None and agent.holding >= 0
            )
            for obj_id in tuple(objects):
                if obj_id < 0 or obj_id not in self._projection_dynamic_ids:
                    continue
                if obj_id in explicit_positive_dynamic_ids:
                    continue
                if obj_id in referenced_dynamic_ids:
                    continue
                objects.pop(obj_id, None)
        if len(self._objects_by_id_cache) >= self._state_cache_limit:
            self._objects_by_id_cache.clear()
        self._objects_by_id_cache[state] = objects
        return objects

    def _role_core_origin_id(
        self,
        obj: CookingZooObjectState,
    ) -> int | None:
        if obj.object_id >= 0:
            return obj.object_id
        key = self._role_core_keys_by_object_id.get(obj.object_id)
        if key is None:
            return None
        for part in key:
            if not part.startswith("origin="):
                continue
            try:
                return int(part.removeprefix("origin="))
            except ValueError:
                return None
        return None

    def _append_content(
        self,
        obj: CookingZooObjectState,
        content_id: int,
    ) -> CookingZooObjectState:
        if content_id in obj.content:
            return obj
        return self._replace_object(obj, content=(*obj.content, content_id))

    def _materialize_released_object(
        self,
        obj: CookingZooObjectState,
        objects: dict[int, CookingZooObjectState],
    ) -> CookingZooObjectState:
        if obj.object_id >= 0 or not self._is_dynamic_kind(obj.kind):
            return obj
        preferred_origin_id = self._role_core_origin_id(obj)
        if preferred_origin_id is not None:
            preferred = self._projection_base_objects_by_id.get(preferred_origin_id)
            if (
                preferred is not None
                and self._is_dynamic_kind(preferred.kind)
                and preferred.kind == obj.kind
            ):
                candidate = preferred
            else:
                candidate = None
        else:
            candidate = None
        occupied_positive_ids = {
            existing.object_id
            for existing in objects.values()
            if existing.object_id >= 0
        }
        referenced_positive_ids = {
            content_id
            for holder in objects.values()
            for content_id in holder.content
            if content_id >= 0
        }
        if candidate is None:
            candidate = next(
                (
                    base_obj
                    for base_obj in self._projection_base_objects
                    if self._is_dynamic_kind(base_obj.kind)
                    and base_obj.kind == obj.kind
                    and base_obj.object_id not in occupied_positive_ids
                    and base_obj.object_id not in referenced_positive_ids
                ),
                None,
            )
        if candidate is None:
            candidate = next(
                (
                    base_obj
                    for base_obj in self._projection_base_objects
                    if self._is_dynamic_kind(base_obj.kind)
                    and base_obj.kind == obj.kind
                ),
                None,
            )
        if candidate is None:
            return obj
        materialized = replace(obj, object_id=candidate.object_id)
        objects.pop(obj.object_id, None)
        for holder_id, holder in tuple(objects.items()):
            if materialized.object_id not in holder.content:
                continue
            objects[holder_id] = self._remove_content(holder, materialized.object_id)
        objects[materialized.object_id] = materialized
        return materialized

    def _remove_content(
        self,
        obj: CookingZooObjectState,
        content_id: int,
    ) -> CookingZooObjectState:
        return self._replace_object(
            obj,
            content=tuple(existing for existing in obj.content if existing != content_id),
        )

    def _after_content_removed(
        self,
        obj: CookingZooObjectState,
    ) -> CookingZooObjectState:
        if obj.content:
            return obj
        if obj.kind in {"Cutboard", "Blender"}:
            return self._replace_attr(obj, "status", "NOT_USABLE")
        return obj

    def _replace_attr(
        self,
        obj: CookingZooObjectState,
        attr_name: str,
        value: str,
    ) -> CookingZooObjectState:
        attrs = dict(obj.attrs)
        attrs[attr_name] = value
        return self._replace_object(obj, attrs=tuple(sorted(attrs.items())))

    def _replace_object(
        self,
        obj: CookingZooObjectState,
        *,
        location: tuple[int, int] | None = None,
        attrs: tuple[tuple[str, str], ...] | None = None,
        content: tuple[int, ...] | None = None,
        free: bool | None = None,
    ) -> CookingZooObjectState:
        next_location = obj.location if location is None else location
        next_attrs = obj.attrs if attrs is None else attrs
        next_content = obj.content if content is None else content
        next_free = obj.free if free is None else free
        if (
            next_location == obj.location
            and next_attrs == obj.attrs
            and next_content == obj.content
            and next_free == obj.free
        ):
            return obj
        return CookingZooObjectState(
            object_id=obj.object_id,
            kind=obj.kind,
            location=next_location,
            attrs=next_attrs,
            content=next_content,
            free=next_free,
        )

    def _stable_value(self, value: Any) -> str:
        if isinstance(value, Enum):
            return value.name
        return str(value)

    def _label_token(self, value: str) -> str:
        chars = [
            char.lower() if char.isalnum() else "_"
            for char in value
        ]
        return "".join(chars).strip("_")

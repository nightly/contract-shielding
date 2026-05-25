import functools
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .assets import LINEAR


# Grid layers
_LAYER_AGENTS = 0
_LAYER_WALLS = 1
_LAYER_DOORS = 2
_LAYER_PLATES = 3
_LAYER_GOAL = 4


class Actions(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Noop = 4


class Entity:
    def __init__(self, entity_id, x, y):
        self.id = entity_id
        self.x = x
        self.y = y


class Agent(Entity):
    pass


class Plate(Entity):
    def __init__(self, entity_id, x, y):
        super().__init__(entity_id, x, y)
        self.pressed = False


class Door(Entity):
    def __init__(self, entity_id, x, y):
        super().__init__(entity_id, x, y)
        self.open = False


class Wall(Entity):
    pass


class Goal(Entity):
    def __init__(self, entity_id, x, y):
        super().__init__(entity_id, x, y)
        self.achieved = False


def env(**kwargs):
    """
    AEC-style PettingZoo env.
    """
    render_mode = kwargs.get("render_mode", None)
    internal_render_mode = "human" if render_mode == "ansi" else render_mode

    aec_env = raw_env(**{**kwargs, "render_mode": internal_render_mode})

    if render_mode == "ansi":
        aec_env = wrappers.CaptureStdoutWrapper(aec_env)

    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def raw_env(**kwargs):
    """
    Convert the simultaneous-action ParallelEnv into an AEC env.
    """
    return parallel_to_aec(parallel_env(**kwargs))


def parallel_env(**kwargs):
    return PressurePlateParallelEnv(**kwargs)


class PressurePlateParallelEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pressureplate_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        height,
        width,
        n_agents,
        sensor_range,
        layout="linear",
        render_mode=None,
        max_cycles=None,
        success_reward=0.0,
        timestep_penalty=0.0,
    ):
        self.grid_size = (height, width)
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.sensor_range = sensor_range
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.success_reward = float(success_reward)
        self.timestep_penalty = float(timestep_penalty)

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        self.agents = []

        self.grid = np.zeros((5, height, width), dtype=np.float32)

        self._view_size = 2 * (sensor_range // 2) + 1
        self._obs_dim = self._view_size * self._view_size * 4 + 2
        self._action_spaces = {
            agent: spaces.Discrete(len(Actions)) for agent in self.possible_agents
        }

        obs_low = np.zeros(self._obs_dim, dtype=np.float32)
        obs_high = np.ones(self._obs_dim, dtype=np.float32)
        obs_high[-2] = width - 1
        obs_high[-1] = height - 1
        self._observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.agent_entities = []
        self.plates = []
        self.walls = []
        self.doors = []
        self.goal = None

        self._rendering_initialized = False
        self.viewer = None

        if layout != "linear":
            raise ValueError(f"Unsupported layout: {layout}")

        if n_agents == 3:
            self.layout = LINEAR["THREE_PLAYERS"]
        elif n_agents == 4:
            self.layout = LINEAR["FOUR_PLAYERS"]
        elif n_agents == 5:
            self.layout = LINEAR["FIVE_PLAYERS"]
        elif n_agents == 6:
            self.layout = LINEAR["SIX_PLAYERS"]
        else:
            raise ValueError(f"Number of agents given ({n_agents}) is not supported.")

        # Same normalization constant the original env used.
        self.max_dist = np.linalg.norm(np.array([0, 0]) - np.array([2, 8]), ord=1)

        self.room_boundaries = np.unique(np.array(self.layout["WALLS"])[:, 1]).tolist()[::-1]
        self.room_boundaries.append(-1)

        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.num_moves = 0
        self._last_hold_door_violations: frozenset[int] = frozenset()
        self._last_wait_door_violations: frozenset[tuple[int, int]] = frozenset()
        self._last_runner_crossing_doors: frozenset[int] = frozenset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self._last_hold_door_violations = frozenset()
        self._last_wait_door_violations = frozenset()
        self._last_runner_crossing_doors = frozenset()
        self._build_world()

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        start_positions = tuple((agent.x, agent.y) for agent in self.agent_entities)
        start_open_doors = tuple(bool(door.open) for door in self.doors)

        # Randomized execution order, same intent as the original env.
        move_order = list(self.np_random.permutation(self.n_agents))

        for i in move_order:
            agent_name = self.possible_agents[i]
            if agent_name not in current_agents:
                continue

            action = actions.get(agent_name, Actions.Noop)
            entity = self.agent_entities[i]
            proposed_pos = [entity.x, entity.y]

            if action == Actions.Up:
                proposed_pos[1] -= 1
                if not self._detect_collision(proposed_pos, moving_agent_id=i):
                    entity.y -= 1

            elif action == Actions.Down:
                proposed_pos[1] += 1
                if not self._detect_collision(proposed_pos, moving_agent_id=i):
                    entity.y += 1

            elif action == Actions.Left:
                proposed_pos[0] -= 1
                if not self._detect_collision(proposed_pos, moving_agent_id=i):
                    entity.x -= 1

            elif action == Actions.Right:
                proposed_pos[0] += 1
                if not self._detect_collision(proposed_pos, moving_agent_id=i):
                    entity.x += 1

            elif action == Actions.Noop:
                pass
            else:
                raise ValueError(f"Invalid action {action} for {agent_name}")

        self._update_plates_and_doors()
        self._sync_dynamic_layers()

        got_goal = any(
            (agent.x == self.goal.x and agent.y == self.goal.y)
            for agent in self.agent_entities
        )
        if got_goal:
            self.goal.achieved = True
        end_positions = tuple((agent.x, agent.y) for agent in self.agent_entities)
        self._last_runner_crossing_doors = self._runner_crossing_doors(
            start_positions,
            end_positions,
        )
        (
            self._last_hold_door_violations,
            self._last_wait_door_violations,
        ) = self._protocol_violations(
            start_positions=start_positions,
            end_positions=end_positions,
            start_open_doors=start_open_doors,
            actions=actions,
            got_goal=got_goal,
        )

        self.num_moves += 1
        env_truncation = (
            self.max_cycles is not None and self.num_moves >= self.max_cycles
        )

        observations = {agent: self.observe(agent) for agent in current_agents}
        rewards = self._get_rewards_dict(current_agents, got_goal=got_goal)
        terminations = {agent: got_goal for agent in current_agents}
        truncations = {agent: env_truncation for agent in current_agents}
        infos = {
            agent: {
                "holds_door_ok": (
                    self.agent_name_mapping[agent]
                    not in self._last_hold_door_violations
                ),
                "waits_for_doors_ok": not any(
                    runner_idx == self.agent_name_mapping[agent]
                    for runner_idx, _ in self._last_wait_door_violations
                ),
            }
            for agent in current_agents
        }

        if got_goal or env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        entity = self.agent_entities[idx]
        return self._get_obs_for_entity(entity)

    def state(self):
        """
        Global state for CTDE-style methods.
        """
        self._sync_dynamic_layers()
        coords = np.array(
            [coord for agent in self.agent_entities for coord in (agent.x, agent.y)],
            dtype=np.float32,
        )
        return np.concatenate((self.grid.reshape(-1), coords), axis=0)

    def render(self):
        if self.render_mode is None:
            return None

        if not self._rendering_initialized:
            self._init_render()

        # Keep rendering.py unchanged by giving it the entity view it expects.
        render_env = SimpleNamespace(
            agents=self.agent_entities,
            walls=self.walls,
            doors=self.doors,
            plates=self.plates,
            goal=self.goal,
        )
        return self.viewer.render(render_env, self.render_mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._rendering_initialized = False

    # -------------------------
    # Internal helpers
    # -------------------------

    def _build_world(self):
        self.grid = np.zeros((5, self.height, self.width), dtype=np.float32)

        # Agents
        self.agent_entities = []
        for i in range(self.n_agents):
            x, y = self.layout["AGENTS"][i]
            self.agent_entities.append(Agent(i, x, y))

        # Walls
        self.walls = []
        for i, (x, y) in enumerate(self.layout["WALLS"]):
            self.walls.append(Wall(i, x, y))
            self.grid[_LAYER_WALLS, y, x] = 1.0

        # Doors
        self.doors = []
        for i, door in enumerate(self.layout["DOORS"]):
            xs, ys = door
            self.doors.append(Door(i, xs, ys))

        # Plates
        self.plates = []
        for i, (x, y) in enumerate(self.layout["PLATES"]):
            self.plates.append(Plate(i, x, y))
            self.grid[_LAYER_PLATES, y, x] = 1.0

        # Goal
        gx, gy = self.layout["GOAL"][0]
        self.goal = Goal("goal", gx, gy)
        self.grid[_LAYER_GOAL, gy, gx] = 1.0

        self._update_plates_and_doors()
        self._sync_dynamic_layers()

    def _sync_dynamic_layers(self):
        # Agents layer
        self.grid[_LAYER_AGENTS, :, :] = 0.0
        for agent in self.agent_entities:
            self.grid[_LAYER_AGENTS, agent.y, agent.x] = 1.0

        # Doors layer: only closed doors should block and appear as obstacles.
        self.grid[_LAYER_DOORS, :, :] = 0.0
        for door in self.doors:
            if not door.open:
                for x, y in zip(door.x, door.y):
                    self.grid[_LAYER_DOORS, y, x] = 1.0

    def _update_plates_and_doors(self):
        for i, plate in enumerate(self.plates):
            assigned_agent = self.agent_entities[plate.id]
            pressed = (plate.x == assigned_agent.x and plate.y == assigned_agent.y)
            plate.pressed = pressed
            self.doors[plate.id].open = pressed

    def _detect_collision(self, proposed_position, moving_agent_id):
        x, y = proposed_position

        # Grid edge
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True

        # Walls
        for wall in self.walls:
            if proposed_position == [wall.x, wall.y]:
                return True

        # Closed doors
        for door in self.doors:
            if not door.open:
                for dx, dy in zip(door.x, door.y):
                    if proposed_position == [dx, dy]:
                        return True

        # Other agents
        for agent in self.agent_entities:
            if agent.id != moving_agent_id and proposed_position == [agent.x, agent.y]:
                return True

        return False

    def _protocol_violations(
        self,
        *,
        start_positions,
        end_positions,
        start_open_doors,
        actions,
        got_goal,
    ):
        hold_violations: set[int] = set()
        wait_violations: set[tuple[int, int]] = set()
        runner_idx = self.n_agents - 1

        for door_idx, plate in enumerate(self.plates):
            if door_idx >= len(start_positions):
                continue
            if start_positions[door_idx] != (plate.x, plate.y):
                continue
            if end_positions[door_idx] == (plate.x, plate.y):
                continue
            if not got_goal:
                hold_violations.add(door_idx)

        runner_action = int(actions.get(f"agent_{runner_idx}", Actions.Noop))
        runner_start = start_positions[runner_idx]
        runner_proposed = self._propose_move(runner_start, runner_action)
        for door_idx, door in enumerate(self.doors):
            if bool(start_open_doors[door_idx]):
                continue
            door_cells = tuple(zip(door.x, door.y))
            if runner_proposed in door_cells:
                wait_violations.add((runner_idx, door_idx))

        return frozenset(hold_violations), frozenset(wait_violations)

    def _runner_crossing_doors(self, start_positions, end_positions) -> frozenset[int]:
        runner_idx = self.n_agents - 1
        if runner_idx >= len(start_positions) or runner_idx >= len(end_positions):
            return frozenset()

        start = start_positions[runner_idx]
        end = end_positions[runner_idx]
        if start == end:
            return frozenset()

        crossing_doors: set[int] = set()
        for door_idx, door in enumerate(self.doors):
            door_cells = frozenset(zip(door.x, door.y))
            door_row = min(door.y)
            moved_into_door = start not in door_cells and end in door_cells
            moved_to_far_side = start in door_cells and end[1] < door_row
            crossed_boundary = start[1] > door_row and end[1] < door_row
            if moved_into_door or moved_to_far_side or crossed_boundary:
                crossing_doors.add(door_idx)
        return frozenset(crossing_doors)

    def _propose_move(self, position, action):
        x, y = position
        if int(action) == int(Actions.Up):
            return (x, y - 1)
        if int(action) == int(Actions.Down):
            return (x, y + 1)
        if int(action) == int(Actions.Left):
            return (x - 1, y)
        if int(action) == int(Actions.Right):
            return (x + 1, y)
        return (x, y)

    def _get_obs_for_entity(self, agent):
        x, y = agent.x, agent.y
        pad = self.sensor_range // 2

        x_left = max(0, x - pad)
        x_right = min(self.width - 1, x + pad)
        y_up = max(0, y - pad)
        y_down = min(self.height - 1, y + pad)

        x_left_padding = pad - (x - x_left)
        x_right_padding = pad - (x_right - x)
        y_up_padding = pad - (y - y_up)
        y_down_padding = pad - (y_down - y)

        def crop_and_pad(layer, pad_value):
            arr = self.grid[layer, y_up : y_down + 1, x_left : x_right + 1]
            arr = np.concatenate(
                (np.full((arr.shape[0], x_left_padding), pad_value), arr), axis=1
            )
            arr = np.concatenate(
                (arr, np.full((arr.shape[0], x_right_padding), pad_value)), axis=1
            )
            arr = np.concatenate(
                (np.full((y_up_padding, arr.shape[1]), pad_value), arr), axis=0
            )
            arr = np.concatenate(
                (arr, np.full((y_down_padding, arr.shape[1]), pad_value)), axis=0
            )
            return arr.reshape(-1)

        # Preserve the original implementation's observation layout:
        # agents, plates, doors, goal, then (x, y).
        # The upstream repo computes a wall crop but does not concatenate it.
        _agents = crop_and_pad(_LAYER_AGENTS, 0.0)
        _plates = crop_and_pad(_LAYER_PLATES, 0.0)
        _doors = crop_and_pad(_LAYER_DOORS, 0.0)
        _goal = crop_and_pad(_LAYER_GOAL, 0.0)

        obs = np.concatenate(
            (_agents, _plates, _doors, _goal, np.array([x, y], dtype=np.float32)),
            axis=0,
        ).astype(np.float32)

        return obs

    def _get_rewards_dict(self, current_agents, *, got_goal=False):
        rewards = {}

        for agent_name in current_agents:
            i = self.agent_name_mapping[agent_name]
            agent = self.agent_entities[i]

            if i == len(self.agent_entities) - 1:
                target_loc = (self.goal.x, self.goal.y)
            else:
                target_loc = (self.plates[i].x, self.plates[i].y)

            curr_room = self._get_curr_room_reward(agent.y)
            agent_loc = (agent.x, agent.y)

            if i == curr_room:
                reward = -np.linalg.norm(
                    np.array(target_loc) - np.array(agent_loc), ord=1
                ) / self.max_dist
            else:
                reward = -len(self.room_boundaries) + 1 + curr_room

            reward -= self.timestep_penalty
            if got_goal:
                reward += self.success_reward

            rewards[agent_name] = float(reward)

        return rewards

    def _get_curr_room_reward(self, agent_y):
        for i, room_level in enumerate(self.room_boundaries):
            if agent_y > room_level:
                return i
        return len(self.room_boundaries) - 1

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True

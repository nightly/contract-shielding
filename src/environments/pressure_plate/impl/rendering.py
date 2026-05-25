"""
2D rendering of the pressure plate domain.
"""

import math
import os

import numpy as np
import pygame


_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_BACKGROUND = (166, 166, 166)


class Viewer:
    def __init__(self, world_size):
        self.rows, self.cols = world_size
        self.grid_size = 1000 / self.rows
        self.width = int(round(self.cols * self.grid_size + 1))
        self.height = int(round(self.rows * self.grid_size + 1))
        self.tile_size = max(1, int(round(self.grid_size)))

        self.window = None
        self.canvas = None
        self.font = None
        self.isopen = True

        script_dir = os.path.dirname(__file__)
        icons_dir = os.path.join(script_dir, "icons")

        self._asset_paths = {
            "agent": os.path.join(icons_dir, "agent.png"),
            "wall": os.path.join(icons_dir, "brick-wall.png"),
            "door": os.path.join(icons_dir, "spiked-fence.png"),
            "plate_off": os.path.join(icons_dir, "plate_off.png"),
            "plate_on": os.path.join(icons_dir, "plate_on.png"),
            "goal": os.path.join(icons_dir, "chest.png"),
            "goal_open": os.path.join(icons_dir, "open-treasure-chest.png"),
        }
        self._raw_images = {}
        self._scaled_images = {}

    def close(self):
        self.window = None
        self.canvas = None
        self.font = None
        self.isopen = False
        if pygame.get_init():
            pygame.quit()

    def render(self, env, return_rgb_array=False):
        self._ensure_initialized(create_window=not return_rgb_array)
        self._handle_events()

        if self.canvas is None:
            self.canvas = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.canvas.fill(_BACKGROUND)
        self._draw_grid()
        self._draw_walls(env)
        self._draw_doors(env)
        self._draw_plates(env)
        self._draw_goal(env)
        self._draw_players(env)
        self._draw_badges(env)

        if self.window is not None:
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()

        if return_rgb_array:
            arr = pygame.surfarray.array3d(self.canvas)
            return np.transpose(arr, (1, 0, 2)).copy()

        return self.isopen

    def _ensure_initialized(self, create_window):
        if not pygame.get_init():
            pygame.init()

        if self.font is None:
            self.font = pygame.font.SysFont("Times New Roman", 12)

        if not self._raw_images:
            for name, path in self._asset_paths.items():
                self._raw_images[name] = pygame.image.load(path)

        if not self._scaled_images:
            size = (self.tile_size, self.tile_size)
            self._scaled_images = {
                name: pygame.transform.smoothscale(image, size)
                for name, image in self._raw_images.items()
            }

        if self.canvas is None:
            self.canvas = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        if create_window and self.window is None:
            pygame.display.init()
            pygame.display.set_caption("Pressure Plate")
            self.window = pygame.display.set_mode((self.width, self.height))
            self.isopen = True

    def _handle_events(self):
        if self.window is None or not pygame.display.get_init():
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False
                pygame.display.quit()
                self.window = None

    def _cell_position(self, row, col):
        return (
            int(round(col * self.grid_size)),
            int(round(row * self.grid_size)),
        )

    def _draw_grid(self):
        for row in range(self.rows + 1):
            y = int(round(self.grid_size * row))
            pygame.draw.line(self.canvas, _BLACK, (0, y), (self.width, y), 1)
        for col in range(self.cols + 1):
            x = int(round(self.grid_size * col))
            pygame.draw.line(self.canvas, _BLACK, (x, 0), (x, self.height), 1)

    def _draw_sprite(self, image_name, row, col):
        x, y = self._cell_position(row, col)
        self.canvas.blit(self._scaled_images[image_name], (x, y))

    def _draw_players(self, env):
        for player in env.agents:
            self._draw_sprite("agent", player.y, player.x)

    def _draw_walls(self, env):
        for wall in env.walls:
            self._draw_sprite("wall", wall.y, wall.x)

    def _draw_doors(self, env):
        for door in env.doors:
            if door.open:
                continue
            for row, col in zip(door.y, door.x):
                self._draw_sprite("door", row, col)

    def _draw_plates(self, env):
        for plate in env.plates:
            image_name = "plate_on" if plate.pressed else "plate_off"
            self._draw_sprite(image_name, plate.y, plate.x)

    def _draw_goal(self, env):
        image_name = "goal_open" if env.goal.achieved else "goal"
        self._draw_sprite(image_name, env.goal.y, env.goal.x)

    def _draw_badges(self, env):
        for agent in env.agents:
            self._draw_badge(agent.y, agent.x, agent.id)

        for plate in env.plates:
            if not plate.pressed:
                self._draw_badge(plate.y, plate.x, plate.id)

        for door in env.doors:
            if door.open:
                continue
            for row, col in zip(door.y, door.x):
                self._draw_badge(row, col, door.id)

    def _draw_badge(self, row, col, badge_id):
        resolution = 6
        radius = self.grid_size / 5
        center_x = col * self.grid_size + 0.75 * self.grid_size
        center_y = row * self.grid_size + 0.75 * self.grid_size

        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + center_x
            y = radius * math.sin(angle) + center_y
            verts.append((int(round(x)), int(round(y))))

        pygame.draw.polygon(self.canvas, _BLACK, verts)
        pygame.draw.polygon(self.canvas, _WHITE, verts, 1)

        label = self.font.render(str(badge_id), True, _WHITE)
        label_rect = label.get_rect(
            center=(int(round(center_x)), int(round(center_y + 2)))
        )
        self.canvas.blit(label, label_rect)

from __future__ import annotations

import numpy as np


PALETTE = {
    "background": (230, 236, 218),
    "board_shadow": (166, 154, 130),
    "cell_a": (226, 236, 205),
    "cell_b": (216, 229, 196),
    "cell_inner": (239, 244, 224),
    "grid": (148, 158, 125),
    "soil": (168, 119, 73),
    "soil_dark": (108, 77, 52),
    "apple": (220, 68, 62),
    "apple_dark": (143, 47, 47),
    "apple_light": (255, 137, 118),
    "leaf": (72, 149, 84),
    "leaf_dark": (42, 105, 62),
    "stem": (109, 76, 42),
    "badge": (255, 250, 229),
    "text": (35, 34, 30),
    "text_soft": (77, 73, 64),
    "white": (255, 255, 255),
    "black": (22, 21, 19),
}

AGENT_COLORS = [
    (41, 111, 170),
    (225, 132, 52),
    (67, 146, 98),
    (135, 94, 171),
    (48, 149, 162),
    (196, 82, 77),
    (104, 116, 132),
    (176, 123, 45),
    (82, 133, 198),
]


def _mix(first: tuple[int, int, int], second: tuple[int, int, int], weight: float):
    return tuple(
        int(round(first[idx] * (1.0 - weight) + second[idx] * weight))
        for idx in range(3)
    )


def _darken(color: tuple[int, int, int], amount: float = 0.25):
    return _mix(color, PALETTE["black"], amount)


def _lighten(color: tuple[int, int, int], amount: float = 0.35):
    return _mix(color, PALETTE["white"], amount)


class Viewer:
    def __init__(self, *, cell_size: int = 48) -> None:
        self.cell_size = int(cell_size)
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None

    def close(self) -> None:
        if self._screen is not None:
            import pygame

            pygame.display.quit()
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None

    def render(self, env, *, return_rgb_array: bool = False):
        import pygame

        pygame.init()
        pygame.font.init()

        width = env.cols * self.cell_size
        height = env.rows * self.cell_size

        if return_rgb_array:
            surface = pygame.Surface((width, height))
        else:
            if self._screen is None:
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Level-Based Foraging")
                self._clock = pygame.time.Clock()
            surface = self._screen

        if self._font is None:
            self._font = pygame.font.SysFont(
                "arial",
                max(16, self.cell_size // 3),
                bold=True,
            )
            self._small_font = pygame.font.SysFont(
                "arial",
                max(12, self.cell_size // 4),
                bold=True,
            )

        self._draw(surface, env)

        if return_rgb_array:
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(env.metadata.get("render_fps", 5))
        return True

    def _draw(self, surface, env) -> None:
        import pygame

        surface.fill(PALETTE["background"])
        self._draw_board(surface, env)

        food_rows, food_cols = env.field.nonzero()
        for row, col in zip(food_rows, food_cols):
            self._draw_food_icon(
                surface,
                int(row),
                int(col),
                int(env.field[row, col]),
            )

        for index, player in enumerate(env.players):
            if player.position is None:
                continue
            row, col = player.position
            self._draw_agent_icon(
                surface,
                int(row),
                int(col),
                index,
                int(player.level),
            )

    def _cell_rect(self, row: int, col: int):
        import pygame

        return pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _draw_board(self, surface, env) -> None:
        import pygame

        board_rect = pygame.Rect(
            0,
            0,
            env.cols * self.cell_size,
            env.rows * self.cell_size,
        )
        pygame.draw.rect(surface, PALETTE["board_shadow"], board_rect.move(0, 3))

        pad = max(2, self.cell_size // 18)
        radius = max(4, self.cell_size // 7)
        for row in range(env.rows):
            for col in range(env.cols):
                rect = self._cell_rect(row, col)
                fill = (
                    PALETTE["cell_a"]
                    if (row + col) % 2 == 0
                    else PALETTE["cell_b"]
                )
                pygame.draw.rect(surface, fill, rect)
                inner = rect.inflate(-2 * pad, -2 * pad)
                pygame.draw.rect(
                    surface,
                    PALETTE["cell_inner"],
                    inner,
                    border_radius=radius,
                )
                pygame.draw.rect(surface, PALETTE["grid"], rect, width=1)

                tuft_color = _mix(fill, PALETTE["leaf"], 0.25)
                tuft_x = rect.left + int(self.cell_size * 0.18)
                tuft_y = rect.bottom - int(self.cell_size * 0.2)
                pygame.draw.line(
                    surface,
                    tuft_color,
                    (tuft_x, tuft_y),
                    (tuft_x + self.cell_size // 10, tuft_y - self.cell_size // 7),
                    width=max(1, self.cell_size // 32),
                )
                pygame.draw.line(
                    surface,
                    tuft_color,
                    (tuft_x + self.cell_size // 8, tuft_y),
                    (tuft_x + self.cell_size // 14, tuft_y - self.cell_size // 9),
                    width=max(1, self.cell_size // 36),
                )

    def _draw_food_icon(self, surface, row: int, col: int, level: int) -> None:
        import pygame

        rect = self._cell_rect(row, col)
        c = self.cell_size
        center = rect.center

        shadow = pygame.Rect(0, 0, int(c * 0.54), int(c * 0.16))
        shadow.center = (center[0], rect.top + int(c * 0.76))
        pygame.draw.ellipse(surface, (101, 92, 72), shadow)

        apple_rect = pygame.Rect(0, 0, int(c * 0.5), int(c * 0.46))
        apple_rect.center = (center[0], center[1] + int(c * 0.05))
        pygame.draw.ellipse(surface, PALETTE["apple_dark"], apple_rect.move(0, 2))
        pygame.draw.ellipse(surface, PALETTE["apple"], apple_rect)

        left_lobe = (center[0] - int(c * 0.12), center[1] - int(c * 0.04))
        right_lobe = (center[0] + int(c * 0.12), center[1] - int(c * 0.04))
        pygame.draw.circle(surface, PALETTE["apple"], left_lobe, int(c * 0.16))
        pygame.draw.circle(surface, PALETTE["apple"], right_lobe, int(c * 0.16))
        pygame.draw.circle(
            surface,
            PALETTE["apple_dark"],
            left_lobe,
            int(c * 0.16),
            width=max(1, c // 28),
        )
        pygame.draw.circle(
            surface,
            PALETTE["apple_dark"],
            right_lobe,
            int(c * 0.16),
            width=max(1, c // 28),
        )

        shine = pygame.Rect(0, 0, int(c * 0.1), int(c * 0.16))
        shine.center = (center[0] - int(c * 0.12), center[1] - int(c * 0.06))
        pygame.draw.ellipse(surface, PALETTE["apple_light"], shine)

        stem_rect = pygame.Rect(0, 0, max(3, c // 12), int(c * 0.2))
        stem_rect.midbottom = (center[0], center[1] - int(c * 0.17))
        pygame.draw.rect(
            surface,
            PALETTE["stem"],
            stem_rect,
            border_radius=max(2, c // 28),
        )

        leaf_rect = pygame.Rect(0, 0, int(c * 0.24), int(c * 0.13))
        leaf_rect.center = (center[0] + int(c * 0.15), center[1] - int(c * 0.21))
        pygame.draw.ellipse(surface, PALETTE["leaf"], leaf_rect)
        pygame.draw.arc(
            surface,
            PALETTE["leaf_dark"],
            leaf_rect,
            0.1,
            2.9,
            width=max(1, c // 36),
        )

        badge_center = (rect.right - int(c * 0.2), rect.top + int(c * 0.22))
        self._draw_badge(
            surface,
            str(level),
            badge_center,
            PALETTE["text"],
            PALETTE["badge"],
            outline=PALETTE["apple_dark"],
        )

    def _draw_agent_icon(
        self,
        surface,
        row: int,
        col: int,
        index: int,
        level: int,
    ) -> None:
        import pygame

        rect = self._cell_rect(row, col)
        c = self.cell_size
        color = AGENT_COLORS[index % len(AGENT_COLORS)]
        outline = _darken(color, 0.35)
        highlight = _lighten(color, 0.28)
        center = rect.center

        shadow = pygame.Rect(0, 0, int(c * 0.58), int(c * 0.16))
        shadow.center = (center[0], rect.top + int(c * 0.78))
        pygame.draw.ellipse(surface, (92, 85, 68), shadow)

        pack_rect = pygame.Rect(0, 0, int(c * 0.24), int(c * 0.38))
        pack_rect.center = (center[0] - int(c * 0.22), center[1] + int(c * 0.05))
        pygame.draw.rect(
            surface,
            _darken(color, 0.15),
            pack_rect,
            border_radius=max(4, c // 9),
        )
        pygame.draw.rect(
            surface,
            outline,
            pack_rect,
            width=max(1, c // 30),
            border_radius=max(4, c // 9),
        )

        body_rect = pygame.Rect(0, 0, int(c * 0.56), int(c * 0.58))
        body_rect.center = (center[0], center[1] + int(c * 0.04))
        pygame.draw.ellipse(surface, outline, body_rect.move(0, 2))
        pygame.draw.ellipse(surface, color, body_rect)
        pygame.draw.arc(
            surface,
            highlight,
            body_rect.inflate(-c // 8, -c // 8),
            3.6,
            5.35,
            width=max(2, c // 18),
        )

        face_rect = pygame.Rect(0, 0, int(c * 0.36), int(c * 0.2))
        face_rect.center = (center[0], center[1] - int(c * 0.03))
        pygame.draw.rect(
            surface,
            (238, 231, 209),
            face_rect,
            border_radius=max(4, c // 8),
        )
        pygame.draw.rect(
            surface,
            outline,
            face_rect,
            width=max(1, c // 32),
            border_radius=max(4, c // 8),
        )

        eye_y = face_rect.centery
        eye_radius = max(1, c // 28)
        pygame.draw.circle(
            surface,
            PALETTE["text"],
            (face_rect.left + int(c * 0.12), eye_y),
            eye_radius,
        )
        pygame.draw.circle(
            surface,
            PALETTE["text"],
            (face_rect.right - int(c * 0.12), eye_y),
            eye_radius,
        )
        smile = pygame.Rect(0, 0, int(c * 0.16), int(c * 0.1))
        smile.center = (center[0], center[1] + int(c * 0.05))
        pygame.draw.arc(
            surface,
            PALETTE["text_soft"],
            smile,
            0.1,
            3.05,
            width=max(1, c // 36),
        )

        antenna_top = (center[0] + int(c * 0.17), center[1] - int(c * 0.32))
        pygame.draw.line(
            surface,
            outline,
            (center[0] + int(c * 0.1), center[1] - int(c * 0.24)),
            antenna_top,
            width=max(1, c // 28),
        )
        pygame.draw.circle(surface, PALETTE["badge"], antenna_top, max(3, c // 14))
        pygame.draw.circle(
            surface,
            outline,
            antenna_top,
            max(3, c // 14),
            width=max(1, c // 36),
        )

        ribbon = pygame.Rect(0, 0, int(c * 0.36), int(c * 0.18))
        ribbon.midbottom = (center[0], rect.bottom - int(c * 0.1))
        pygame.draw.rect(surface, outline, ribbon, border_radius=max(3, c // 12))
        label_surface = self._small_font.render(f"A{index}", True, PALETTE["white"])
        surface.blit(label_surface, label_surface.get_rect(center=ribbon.center))

        badge_center = (rect.right - int(c * 0.18), rect.bottom - int(c * 0.2))
        self._draw_badge(
            surface,
            str(level),
            badge_center,
            PALETTE["text"],
            PALETTE["badge"],
            outline=outline,
        )

    def _draw_badge(
        self,
        surface,
        text: str,
        center,
        text_color,
        fill_color,
        *,
        outline=PALETTE["text"],
    ) -> None:
        import pygame

        radius = max(9, self.cell_size // 5)
        pygame.draw.circle(
            surface,
            (99, 90, 72),
            (center[0], center[1] + 2),
            radius,
        )
        pygame.draw.circle(surface, fill_color, center, radius)
        pygame.draw.circle(
            surface,
            outline,
            center,
            radius,
            width=max(2, self.cell_size // 28),
        )
        label_surface = self._small_font.render(text, True, text_color)
        surface.blit(label_surface, label_surface.get_rect(center=center))

from __future__ import annotations

import random
from dataclasses import dataclass

import pygame

from .game import GuessResult, InvalidGuess, LetterState, WordleGame, keyboard_state
from .players import PLAYER_MODES, HumanPlayer, Player, make_player

# --- Layout ---
WINDOW_W, WINDOW_H = 500, 760
HEADER_H = 60

TILE_SIZE = 62
TILE_GAP = 6
GRID_TOP = HEADER_H + 24
GRID_W = 5 * TILE_SIZE + 4 * TILE_GAP
GRID_LEFT = (WINDOW_W - GRID_W) // 2

KEY_W, KEY_H = 38, 50
KEY_GAP = 5
KEYBOARD_TOP = GRID_TOP + 6 * (TILE_SIZE + TILE_GAP) + 28
KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")  # bottom row gets ENTER/BACK

# --- Colors (Wordle-ish palette) ---
BG = (255, 255, 255)
HEADER_BG = (255, 255, 255)
HEADER_LINE = (211, 214, 218)
TEXT = (26, 26, 27)
TEXT_MUTED = (120, 124, 126)
TILE_EMPTY_BORDER = (211, 214, 218)
TILE_FILLED_BORDER = (135, 138, 140)
COLOR_CORRECT = (106, 170, 100)
COLOR_PRESENT = (201, 180, 88)
COLOR_ABSENT = (120, 124, 126)
KEY_BG = (211, 214, 218)
KEY_TEXT = (26, 26, 27)
TILE_TEXT_DARK = (26, 26, 27)
TILE_TEXT_LIGHT = (255, 255, 255)
TOAST_BG = (26, 26, 27)
BUTTON_BG = (240, 240, 240)
BUTTON_BG_ACTIVE = (106, 170, 100)
BUTTON_BORDER = (180, 180, 180)
MENU_DISABLED_BG = (245, 245, 245)
MENU_ITEM_H = 28

STATE_COLORS = {
    LetterState.CORRECT: COLOR_CORRECT,
    LetterState.PRESENT: COLOR_PRESENT,
    LetterState.ABSENT: COLOR_ABSENT,
}

REVEAL_PER_TILE_MS = 250
SHAKE_DURATION_MS = 350
TOAST_DURATION_MS = 1500
# Wait for tile reveal before the solver submits its next guess.
MODEL_GUESS_PAUSE_MS = 400


@dataclass
class Toast:
    text: str
    expires_at: int


@dataclass
class Shake:
    started_at: int
    row_index: int


class WordleUI:
    def __init__(self, answers: list[str], allowed_words: set[str]) -> None:
        pygame.init()
        pygame.display.set_caption("Wordle")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("helveticaneue,arial", 28, bold=True)
        self.font_tile = pygame.font.SysFont("helveticaneue,arial", 32, bold=True)
        self.font_key = pygame.font.SysFont("helveticaneue,arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("helveticaneue,arial", 14)
        self.font_status = pygame.font.SysFont("helveticaneue,arial", 16, bold=True)

        self.answers = answers
        self.allowed_words = allowed_words

        self.human = HumanPlayer()
        self.active_player: Player = self.human
        self.game: WordleGame = self._new_game()

        self.last_submit_at: int | None = None
        self.shake: Shake | None = None
        self.toast: Toast | None = None

        self.selected_player_id = "human"
        self.player_menu_open = False
        self.dropdown_rect = pygame.Rect(WINDOW_W - 168, 14, 153, 32)
        self._menu_hitboxes: list[tuple[pygame.Rect, tuple[str, str, bool]]] = []
        self._solver_players: dict[str, Player] = {}
        self._key_rects: list[tuple[pygame.Rect, str]] = []

    # --- Game lifecycle ---
    def _new_game(self) -> WordleGame:
        answer = random.choice(self.answers)
        self.human.reset()
        self.last_submit_at = None
        self.shake = None
        return WordleGame(answer, allowed_words=self.allowed_words)

    def reset(self) -> None:
        self.game = self._new_game()
        self._toast(f"New game — good luck!")

    def _toast(self, text: str) -> None:
        self.toast = Toast(text=text, expires_at=pygame.time.get_ticks() + TOAST_DURATION_MS)

    def set_player(self, mode_id: str) -> None:
        mode = next((m for m in PLAYER_MODES if m[0] == mode_id), None)
        if mode is None or not mode[2]:
            raise ValueError(f"Player not available: {mode_id!r}")
        self.selected_player_id = mode_id
        self.player_menu_open = False
        if mode_id == "human":
            self.active_player = self.human
        else:
            if mode_id not in self._solver_players:
                self._solver_players[mode_id] = make_player(mode_id, self.answers)
            self.active_player = self._solver_players[mode_id]

    def _menu_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.dropdown_rect.x,
            self.dropdown_rect.bottom + 2,
            self.dropdown_rect.width,
            len(PLAYER_MODES) * MENU_ITEM_H,
        )

    def _handle_player_dropdown_click(self, pos: tuple[int, int]) -> bool:
        menu_rect = self._menu_rect()
        dropdown_area = self.dropdown_rect.union(menu_rect) if self.player_menu_open else self.dropdown_rect

        if self.player_menu_open:
            for rect, (mode_id, label, available) in self._menu_hitboxes:
                if rect.collidepoint(pos):
                    if available:
                        self.set_player(mode_id)
                        self._toast(
                            f"{label} — watch it play" if mode_id != "human" else "Human mode"
                        )
                    return True
            if not dropdown_area.collidepoint(pos):
                self.player_menu_open = False
            return True

        if self.dropdown_rect.collidepoint(pos):
            self.player_menu_open = True
            return True
        return False

    def _model_ready_to_guess(self) -> bool:
        if self.last_submit_at is None:
            return True
        reveal_ms = WordleGame.WORD_LENGTH * REVEAL_PER_TILE_MS
        elapsed = pygame.time.get_ticks() - self.last_submit_at
        return elapsed >= reveal_ms + MODEL_GUESS_PAUSE_MS

    # --- Main loop ---
    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                # R restarts only when the game has ended; while playing, R is a letter.
                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_r
                    and self.game.is_over
                ):
                    self.reset()
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._handle_player_dropdown_click(event.pos):
                        continue
                    if self.player_menu_open:
                        self.player_menu_open = False
                        continue
                    if self._handle_key_click(event.pos):
                        continue

                if not self.game.is_over:
                    self._dispatch_to_player(event)

            if not self.game.is_over and not isinstance(self.active_player, HumanPlayer):
                if self._model_ready_to_guess():
                    guess = self.active_player.tick(self.game)
                    if guess is not None:
                        self._try_submit(guess)

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _dispatch_to_player(self, event: pygame.event.Event) -> None:
        guess = self.active_player.handle_event(event, self.game)
        if guess is not None:
            self._try_submit(guess)

    def _handle_key_click(self, pos: tuple[int, int]) -> bool:
        for rect, label in self._key_rects:
            if rect.collidepoint(pos):
                if self.game.is_over or not isinstance(self.active_player, HumanPlayer):
                    return True
                guess = self.human.press_key(label)
                if guess is not None:
                    self._try_submit(guess)
                return True
        return False

    def _try_submit(self, guess: str) -> None:
        try:
            self.game.submit(guess)
        except InvalidGuess as e:
            self.shake = Shake(started_at=pygame.time.get_ticks(), row_index=len(self.game.history))
            self._toast(str(e))
            # Restore the buffer so the player can edit it
            if isinstance(self.active_player, HumanPlayer):
                self.active_player._buffer = guess  # noqa: SLF001
            return

        self.last_submit_at = pygame.time.get_ticks()
        if self.game.is_won:
            won_by = (
                f"{self.active_player.name} solved it!"
                if not isinstance(self.active_player, HumanPlayer)
                else "You got it!"
            )
            self._toast(won_by)
        elif self.game.is_lost:
            self._toast(f"The word was: {self.game.answer.upper()}")

    # --- Drawing ---
    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_header()
        self._draw_board()
        self._draw_keyboard()
        self._draw_status()
        self._draw_player_dropdown()
        self._draw_toast()

    def _draw_header(self) -> None:
        pygame.draw.line(self.screen, HEADER_LINE, (0, HEADER_H), (WINDOW_W, HEADER_H), 1)
        title = self.font_title.render("Wordle", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(WINDOW_W // 2, HEADER_H // 2)))

    def _draw_player_dropdown(self) -> None:
        is_model = not isinstance(self.active_player, HumanPlayer)
        bg = BUTTON_BG_ACTIVE if is_model else BUTTON_BG
        pygame.draw.rect(self.screen, bg, self.dropdown_rect, border_radius=6)
        pygame.draw.rect(self.screen, BUTTON_BORDER, self.dropdown_rect, 1, border_radius=6)

        label = f"Player: {self.active_player.name}"
        text_color = (255, 255, 255) if is_model else TEXT
        text = self.font_small.render(label, True, text_color)
        text_rect = text.get_rect(midleft=(self.dropdown_rect.x + 10, self.dropdown_rect.centery))
        self.screen.blit(text, text_rect)

        arrow = self.font_small.render("▼" if not self.player_menu_open else "▲", True, text_color)
        self.screen.blit(
            arrow,
            arrow.get_rect(midright=(self.dropdown_rect.right - 8, self.dropdown_rect.centery)),
        )

        if not self.player_menu_open:
            return

        menu_rect = self._menu_rect()
        pygame.draw.rect(self.screen, BG, menu_rect, border_radius=6)
        pygame.draw.rect(self.screen, BUTTON_BORDER, menu_rect, 1, border_radius=6)

        self._menu_hitboxes = []
        for i, mode in enumerate(PLAYER_MODES):
            mode_id, mode_label, available = mode
            item_rect = pygame.Rect(
                menu_rect.x,
                menu_rect.y + i * MENU_ITEM_H,
                menu_rect.width,
                MENU_ITEM_H,
            )
            self._menu_hitboxes.append((item_rect, mode))

            if mode_id == self.selected_player_id:
                item_bg = BUTTON_BG_ACTIVE
                item_color = (255, 255, 255)
            elif available:
                item_bg = BUTTON_BG
                item_color = TEXT
            else:
                item_bg = MENU_DISABLED_BG
                item_color = TEXT_MUTED

            pygame.draw.rect(self.screen, item_bg, item_rect)
            if i > 0:
                pygame.draw.line(
                    self.screen,
                    BUTTON_BORDER,
                    (item_rect.left + 6, item_rect.top),
                    (item_rect.right - 6, item_rect.top),
                    1,
                )

            suffix = "" if available else " (soon)"
            item_text = self.font_small.render(mode_label + suffix, True, item_color)
            self.screen.blit(
                item_text,
                item_text.get_rect(midleft=(item_rect.x + 10, item_rect.centery)),
            )

    def _draw_board(self) -> None:
        history = self.game.history
        current_row_index = len(history)
        in_progress = self.active_player.current_input

        now = pygame.time.get_ticks()
        shake_dx = self._shake_offset(now)

        for row in range(self.game.max_guesses):
            is_current_row = row == current_row_index
            row_dx = shake_dx if (self.shake and self.shake.row_index == row) else 0

            for col in range(WordleGame.WORD_LENGTH):
                x = GRID_LEFT + col * (TILE_SIZE + TILE_GAP) + row_dx
                y = GRID_TOP + row * (TILE_SIZE + TILE_GAP)
                rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

                if row < current_row_index:
                    self._draw_revealed_tile(rect, history[row], col, row)
                elif is_current_row and col < len(in_progress):
                    self._draw_filled_tile(rect, in_progress[col].upper())
                else:
                    self._draw_empty_tile(rect)

    def _shake_offset(self, now: int) -> int:
        if not self.shake:
            return 0
        elapsed = now - self.shake.started_at
        if elapsed > SHAKE_DURATION_MS:
            self.shake = None
            return 0
        # Damped sine wobble
        import math
        amplitude = 8 * (1 - elapsed / SHAKE_DURATION_MS)
        return int(amplitude * math.sin(elapsed / 30.0))

    def _draw_empty_tile(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, TILE_EMPTY_BORDER, rect, 2)

    def _draw_filled_tile(self, rect: pygame.Rect, letter: str) -> None:
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, TILE_FILLED_BORDER, rect, 2)
        text = self.font_tile.render(letter, True, TILE_TEXT_DARK)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_revealed_tile(self, rect: pygame.Rect, result: GuessResult, col: int, row: int) -> None:
        # Stagger reveal for the most recent submission only.
        revealed = True
        if self.last_submit_at is not None and row == len(self.game.history) - 1:
            elapsed = pygame.time.get_ticks() - self.last_submit_at
            revealed = elapsed >= (col + 1) * REVEAL_PER_TILE_MS

        letter = result.word[col].upper()
        if not revealed:
            self._draw_filled_tile(rect, letter)
            return

        color = STATE_COLORS[result.states[col]]
        pygame.draw.rect(self.screen, color, rect)
        text = self.font_tile.render(letter, True, TILE_TEXT_LIGHT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_keyboard(self) -> None:
        states = keyboard_state(self.game.history)
        y = KEYBOARD_TOP
        self._key_rects = []

        for row_index, row in enumerate(KEYBOARD_ROWS):
            keys: list[tuple[str, int]] = [(c, KEY_W) for c in row]
            if row_index == 2:
                keys = [("ENTER", 60)] + keys + [("BACK", 60)]

            row_w = sum(w for _, w in keys) + KEY_GAP * (len(keys) - 1)
            x = (WINDOW_W - row_w) // 2

            for label, w in keys:
                rect = pygame.Rect(x, y, w, KEY_H)
                state = states.get(label.lower()) if len(label) == 1 else None

                if state is not None:
                    bg = STATE_COLORS[state]
                    text_color = TILE_TEXT_LIGHT
                else:
                    bg = KEY_BG
                    text_color = KEY_TEXT

                pygame.draw.rect(self.screen, bg, rect, border_radius=4)
                text = self.font_key.render(label, True, text_color)
                self.screen.blit(text, text.get_rect(center=rect.center))
                self._key_rects.append((rect, label))
                x += w + KEY_GAP

            y += KEY_H + KEY_GAP

    def _draw_status(self) -> None:
        if self.game.is_won:
            msg = "You won! Press R for a new game."
        elif self.game.is_lost:
            msg = f"The word was {self.game.answer.upper()}. Press R to retry."
        else:
            msg = f"{self.game.guesses_left} guesses left · Esc: quit"

        text = self.font_status.render(msg, True, TEXT_MUTED)
        self.screen.blit(text, text.get_rect(center=(WINDOW_W // 2, WINDOW_H - 18)))

    def _draw_toast(self) -> None:
        if not self.toast:
            return
        if pygame.time.get_ticks() > self.toast.expires_at:
            self.toast = None
            return

        text = self.font_status.render(self.toast.text, True, (255, 255, 255))
        padding = 14
        rect = text.get_rect()
        rect.inflate_ip(padding * 2, padding)
        rect.center = (WINDOW_W // 2, HEADER_H + 6 + rect.height // 2)
        pygame.draw.rect(self.screen, TOAST_BG, rect, border_radius=6)
        self.screen.blit(text, text.get_rect(center=rect.center))

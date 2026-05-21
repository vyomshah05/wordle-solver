# Wordle Visual Simulator (Python + pygame)

## Context

The repo is greenfield — only a one-line README and a stock Python `.gitignore`. The end goal is an LLM that solves Wordle on its own, but step one is a visual simulator of the game itself. The simulator must:

- Mimic the real Wordle UI (5-letter word, 6 guesses, gray/yellow/green tile feedback).
- Let a human play with the keyboard.
- Expose a clean **Player** plug-in seam so an LLM-driven solver can be dropped in later without touching the UI or game-rules code.

Deliberately out of scope for this pass: actually wiring up a model, hard mode, dark mode toggle, animations beyond a basic flip/reveal, stats/persistence, daily-word logic.

## Approach

Three-layer split so the future LLM player only depends on the game-logic layer:

1. **Game logic** — pure-Python rules, no I/O. Easy to unit test and to feed to a solver.
2. **Player interface** — abstract base class + `HumanPlayer` (event-driven, keyboard) and a `ModelPlayer` stub that raises `NotImplementedError` until step two.
3. **UI** — pygame window that renders the board + on-screen keyboard and dispatches events to whichever player is active. An on-screen header button toggles between human and model mode so both can coexist.

### File layout

```
wordle-solver/
├── main.py                      # entry point: parses --player flag, starts UI loop
├── requirements.txt             # pygame
├── wordle/
│   ├── __init__.py
│   ├── game.py                  # WordleGame, GuessResult, LetterState (Enum)
│   ├── players.py               # Player (ABC), HumanPlayer, ModelPlayer (stub)
│   ├── ui.py                    # pygame rendering + event loop
│   └── words.py                 # load_words()
└── data/
    └── allowed.txt              # ~13,000 valid 5-letter words (any one can be the answer)
```

### Game logic (`wordle/game.py`)

- `LetterState` enum: `CORRECT` (green), `PRESENT` (yellow), `ABSENT` (gray).
- `GuessResult` dataclass: `word: str`, `states: list[LetterState]`.
- `WordleGame`:
  - `__init__(answer, max_guesses=6, allowed_words=None)`
  - `submit(guess) -> GuessResult` — validates length/membership, scores, appends to history.
  - `is_won`, `is_lost`, `is_over`, `history`, `guesses_left`.
  - **Two-pass scoring** for duplicate-letter correctness.
- `keyboard_state(history)` returns best-known state per letter (CORRECT > PRESENT > ABSENT).

### Player seam (`wordle/players.py`)

```python
class Player(ABC):
    def handle_event(self, event, game) -> str | None: ...
```

- `HumanPlayer`: keeps a `current_input` buffer; KEYDOWN appends letters / handles backspace / submits on Enter.
- `ModelPlayer`: stub that ignores events; constructor raises `NotImplementedError`.

### UI (`wordle/ui.py`)

- pygame window ~500×700 px.
- 6×5 tile grid + on-screen QWERTY keyboard tinted from `keyboard_state(...)`.
- Header bar with **"Player: Human / Model" toggle button**; clicking model mode currently shows a "model not wired yet" toast.
- Status text + footer hint "R: new game · Esc: quit".
- Invalid-guess shake (200 ms x-offset). Reveal: per-tile 250 ms staggered color fill.

### Word list

- Full official NYT Wordle guess list in `data/allowed.txt` (~13,000 words).
- Any word in the list can be the secret answer **and** is a valid guess.

### Entry point

```
python main.py                # human play
python main.py --player model # raises until LLM wired
```

## Verification

1. `pip install -r requirements.txt`
2. `python main.py` — type a word, press Enter, watch tiles color and keyboard update.
3. Edge cases: duplicate letters (`ALLEY` vs `LLAMA`), invalid guess shake, win/lose end states, `R` to reset, header toggle button toasts when model picked.

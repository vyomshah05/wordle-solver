# wordle-solver

Wordle simulator with three project models plus human play.

| Model | ID | Code |
|-------|-----|------|
| **Model 1** — constraint + frequency baseline | `base_model` | `base_model/` |
| **Model 2** — entropy solver (primary) | `entropy` | `entropy/` *(planned)* |
| **Model 3** — reinforcement learning | `rl` | `rl/` |

## Run

```bash
pip install -r requirements.txt
python main.py                           # human
python main.py --player base_model       # Model 1
python main.py --player rl               # Model 3
python benchmark.py --limit 500          # benchmark Model 1
python -m rl.evaluate --limit 500        # benchmark Model 3 (see rl/README.md)
```

Use the **Player** dropdown (top-right) to switch during a session.

## Layout

```
main.py
benchmark.py
wordle/                 # game + UI
  game.py  words.py  players.py  ui.py
base_model/             # Model 1
entropy/                # Model 2 (teammate adds this)
rl/                     # Model 3
data/allowed.txt
```

## Adding Model 2 (entropy)

The Model toggle is wired to an information-theoretic entropy solver.

### Entropy solver

- First run builds an N×N pattern matrix (~1 min, ~113 MB) and an opener
  cache; subsequent runs load from `wordle/entropy_solver/cache/` in under
  5 s. Delete the cache folder to force a rebuild.
- Run a full benchmark: `python scripts/run_entropy_simulation.py --sample 500 --seed 42`
- Tests: `pytest tests/`
1. Add an `entropy/` package with solver + optional `Player` class.
2. In `wordle/players.py`, set `("entropy", "Model 2: Entropy", True)` and add a `make_player` branch.

Evaluation story: **random → Model 1 (base) → Model 2 (entropy) → Model 3 (RL)**.

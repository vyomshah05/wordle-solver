# Model 3: Reinforcement Learning Wordle Solver

This folder implements the **RL comparison model** for the Wordle solver project: a learned policy that competes against the constraint+frequency baseline (Model 1) and the entropy solver (Model 2).

The design follows your project spec: **Option A (tabular Q-learning)** is the main, timeline-friendly path; **Option B (numpy DQN sketch)** is included for extension and report discussion.

---

## Why not learn 12,000 actions directly?

The raw RL state “set of remaining candidate words” has astronomically many values. Storing \(Q(s, \text{word})\) for every word at every state is infeasible.

**Option A** compresses the problem:

| Layer | What it represents |
|--------|-------------------|
| **State** | Coarse features: turn, bucketed candidate count, greens, yellows, grays |
| **Abstract action** | A *strategy*: frequency guess, entropy guess, random candidate, probe |
| **Concrete guess** | `heuristics.pick_guess()` turns the strategy into one 5-letter word |

The agent learns *when* to use entropy vs frequency vs probing — not memorizing individual words. That is a realistic inductive bias for a course project and still **uses Wordle color feedback** (via `constraints.filter_candidates`).

**Option B** (`agent_dqn.py`) encodes the board as a fixed vector and scores a sampled subset of legal guesses with a small MLP. It is more expressive but slower to train and tune.

---

## File map

```
rl/
├── constraints.py    # Filter candidates from green/yellow/gray history
├── heuristics.py     # Frequency / entropy / probe guess builders
├── state.py          # TabularStateKey + StateVector for neural policy
├── environment.py    # WordleRLEnv: reset, step, rewards (no pygame)
├── agent_tabular.py  # QLearningAgent + save/load JSON
├── agent_dqn.py      # Optional numpy MLP (Option B)
├── player.py         # RLPlayer → plugs into wordle/ui.py
├── train.py          # Self-play training CLI
├── evaluate.py       # Win rate & mean guesses benchmark
└── checkpoints/    # Saved q_table.json (gitignored)
```

---

## Rewards and episode flow

Each training game is one **episode**:

1. Sample a secret answer from the word list.
2. Repeat until win or 6 guesses:
   - Observe tabular state \(s\)
   - Choose abstract action \(a \in \{\text{FREQUENCY, ENTROPY, RANDOM, PROBE}\}\)
   - Build guess \(g = \text{pick\_guess}(a, \ldots)\)
   - Submit to `WordleGame`, update candidates from feedback
   - Receive reward, observe \(s'\)

**Rewards** (`environment.py`):

| Event | Reward |
|--------|--------|
| Each guess | \(-1\) |
| Win | \(+10\) |
| Loss (6 wrong) | \(-20\) |

This shapes the policy to win quickly and avoid running out of rows.

**Q-learning update** (`agent_tabular.py`):

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \bigr]
\]

- \(\alpha = 0.15\) learning rate  
- \(\gamma = 0.95\) discount  
- \(\epsilon\)-greedy exploration decaying over 30k episodes  

---

## Module-by-module

### `constraints.py`

Uses the same two-pass scoring as `wordle/game.py`. A word stays in `candidates` only if, for every past guess, `score_guess(guess, word)` matches the real feedback pattern.

This is how every solver “listens” to Wordle.

### `heuristics.py`

| `GuessStrategy` | Behavior |
|-----------------|----------|
| `FREQUENCY` | Maximize sum of positional letter frequencies over candidates; penalize repeats |
| `ENTROPY` | Maximize expected partition entropy of candidates (one-step) |
| `RANDOM_CANDIDATE` | Uniform random remaining answer |
| `PROBE` | Entropy over full 13k list (easy mode only; hard mode falls back) |

Opening guess is fixed to **`slate`** until you run offline analysis (compare with Model 2’s SALET/CRANE study).

### `state.py`

**Tabular buckets** keep the Q-table finite:

- Candidates: 1, 2–5, 6–20, 21–100, …, 2000+
- Turn: 0–5
- Greens: 0–5 locked positions
- Yellow / gray letter counts: bucketed

### `environment.py`

Training-only wrapper around `WordleGame`. No rendering — fast self-play loops.

### `player.py`

`RLPlayer` implements the existing `Player` API:

- `tick(game)` returns one guess per row (tracks `len(game.history)` so pygame’s 60 FPS loop does not spam submits).
- Loads `checkpoints/q_table.json` by default.

---

## Commands

Install numpy (added to project `requirements.txt`):

```bash
pip install -r requirements.txt
```

**Train** (start with 50k episodes; more = stabler Q-table):

```bash
python -m rl.train --episodes 50000
```

**Evaluate**:

```bash
python -m rl.evaluate --games 2000
```

**Play in the UI**:

```bash
python main.py --player rl
```

**Easy mode** (allow probe strategy over full word list):

```bash
python -m rl.train --episodes 50000 --easy-mode
```

---

## Hybrid idea (entropy + RL endgame)

For your report, a strong variant without full DQN:

1. Use **entropy** when `len(candidates) > 20`
2. Use **RL tabular policy** when `len(candidates) <= 20`

That matches the spec’s “RL only for endgame” middle ground. You can implement this in `player.py` by branching on `constraint.num_candidates` before `agent.best_action()`.

---

## Expected results & narrative

Honest expectation from the project brief: **RL may not beat entropy** (3.4–3.6 mean guesses). That is a valid finding:

> Information-theoretic search encodes the right bias; RL learns when to switch heuristics but struggles to outperform a near-optimal greedy entropy policy.

Report metrics:

- Win rate within 6 guesses (target 98%+ for tabular, higher with more training)
- Mean guesses on wins
- Strategy histogram from `evaluate.py` (does the agent learn to prefer ENTROPY early and FREQUENCY late?)

---

## Integration with teammates’ models

| Model | Folder (suggested) | Relationship to RL |
|-------|-------------------|---------------------|
| Model 1 | `solvers/frequency.py` (future) | FREQUENCY strategy in RL *is* that baseline |
| Model 2 | `solvers/entropy.py` (future) | ENTROPY strategy in RL *is* that solver |
| Model 3 | `rl/` (this folder) | Learns **mixture policy** over those tools |

Your four-way evaluation table: Random → Frequency → Entropy → RL mixture.

---

## Option B (DQN) next steps

`agent_dqn.py` is a skeleton. To productionize:

1. Fix bootstrap target (max over next-step guess pool).
2. Train 100k+ episodes with replay buffer full.
3. Compare wall-clock vs tabular Q-learning.

For a tight timeline, **ship Option A** and cite Option B as future work unless benchmarks are required.

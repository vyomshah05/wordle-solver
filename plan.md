Ready for review
Select text to add comments on the plan
Entropy-Based Wordle Solver — Implementation Plan
Context
The repo already has a working Wordle simulator (Game logic, pygame UI, Player ABC, word list) with a "Player: Model" UI toggle that currently shows "Model mode (not wired yet)" because ModelPlayer.__init__ raises NotImplementedError. We're filling in that seam with an information-theoretic entropy solver (3Blue1Brown-style), which (a) makes the Model toggle functional and (b) serves as one of three benchmarked solvers for the course report — alongside teammates' constraint+frequency and RL approaches.

The solver must be cleanly importable for head-to-head comparison and run interactively in the pygame UI as well as in a headless simulation script.

Findings from Exploration
A few details differ from the original Claude-chat prompt and shape this plan:

Feedback is LetterState enum, not integers. wordle/game.py:8-17 defines LetterState.{ABSENT,PRESENT,CORRECT} and GuessResult.states: tuple[LetterState, ...]. The solver internally uses base-3 ints (0=gray, 1=yellow, 2=green); a small translation helper bridges them.
score_guess(guess, answer) is already exposed at wordle/game.py:75-96 with correct two-pass duplicate-letter handling. Reuse it in tests as the reference oracle; the pattern matrix itself must still be vectorized in NumPy for speed.
Word list is ~10,656 words (not ~13k as the prompt assumed). Pattern matrix is N×N int8 ≈ 113 MB — fits in RAM, builds in well under 3 min vectorized.
HumanPlayer does not override tick — it returns guesses from handle_event/press_key (see wordle/players.py:33-74). The "one guess per turn, None otherwise" pattern for ModelPlayer.tick must be implemented from scratch, gated on len(game.history) advancing.
UI calls active_player.tick(self.game) at 60 FPS at wordle/ui.py:156-160 and submits any non-None return immediately. Game reset (R key) creates a fresh Game, so len(history) dropping to 0 signals "new round → reset solver."
No tests/, scripts/, results/ dirs exist. requirements.txt only has pygame — must add numpy and pytest.
Files Modified / Created
Created (new):

wordle/entropy_solver/__init__.py — exports EntropySolver
wordle/entropy_solver/patterns.py — feedback encoding + vectorized N×N pattern matrix + hash-keyed cache
wordle/entropy_solver/solver.py — EntropySolver class
wordle/entropy_solver/cache/.gitkeep (cache artifacts written here; not committed)
tests/__init__.py
tests/test_entropy.py
scripts/run_entropy_simulation.py
results/.gitkeep
Modified:

wordle/players.py — replace ModelPlayer stub with real implementation
requirements.txt — add numpy, pytest
README.md — short "Running the entropy solver" section under existing "Plugging in a model"
.gitignore — ignore wordle/entropy_solver/cache/*.npy, *.json, *.hash; ignore results/*.json
Strictly read-only (DO NOT EDIT): wordle/game.py, wordle/ui.py, wordle/words.py, main.py.

Implementation Detail
1. wordle/entropy_solver/patterns.py
encode_states(states: tuple[LetterState, ...]) -> int: maps a 5-tuple of LetterState to a base-3 int in [0, 243) using ABSENT=0, PRESENT=1, CORRECT=2. Inverse decode_pattern(p: int) -> tuple[int, ...] for tests/debug.
pattern_int(guess: str, answer: str) -> int: thin wrapper that calls the existing score_guess from wordle/game.py:75 then encode_states. Used as the reference oracle in tests, never in the hot path.
build_pattern_matrix(words: list[str]) -> np.ndarray: vectorized N×N int8 builder.
Encode words to an (N, 5) int8 array of letter codes (a→0…z→25).
For each guess position p (loop of 5), compute equal[p] = guess_letters[:, None, p] == answer_letters[None, :, p] → (N, N) bool. Start pattern = 2 * equal[p] * 3**(4-p) accumulated across positions.
For yellows, use the canonical "count remaining instances of letter c in answer minus greens already credited" trick across the 26 letters: for each letter c, build guess_unmatched_count_c and answer_unmatched_count_c (positions where that letter appears AND wasn't already green), then per (i,j) credit min(guess_unmatched_count, answer_unmatched_count) yellows greedily from left to right in guess order. Implemented vectorized across all (i,j) pairs simultaneously.
Output dtype int8 (values 0–242 fit fine).
Target: full ~10.6k×10.6k matrix in < 60 s on a laptop.
Cache: write to wordle/entropy_solver/cache/pattern_matrix.npy and a sibling pattern_matrix.hash containing sha256 of the sorted, joined word list. On load: hash must match or rebuild.
2. wordle/entropy_solver/solver.py
class EntropySolver:
    def __init__(self, word_list: list[str]) -> None: ...
    def reset(self) -> None: ...                       # restores candidates to full word list
    def choose_guess(self, history: list[GuessResult]) -> str: ...
Accepts list[GuessResult] (the game's native type) — internally maps states→ints. No translation needed at call sites.
On init: load/build pattern matrix; load/compute cached opener at cache/opener.json. When building opener cache from scratch, print top 10 openers + entropies to stdout (for the report).
Candidate set stored as a boolean mask of length N (so pattern_matrix[:, candidates_mask] slicing is cheap).
Filter step: for each (guess, pattern_int) in history, find guess's row index, then candidates_mask &= (pattern_matrix[guess_idx] == pattern_int). O(N) per filter.
Entropy score for guess w:
counts = np.bincount(pattern_matrix[w_idx, candidates_mask], minlength=243)
probs = counts / counts.sum()
H = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
Computed across all guess rows in one pass — broadcast bincount via NumPy is slightly awkward, so loop the bincount but keep the inner work in NumPy. For ~10.6k guesses × ~10.6k candidate columns this is fast enough (< 0.5s after opener).
Tie-break: prefer guesses still in the candidate set (their bit in candidates_mask is set).
Fixed opener: when history == [], return cached opener directly — no recomputation.
Two-step lookahead when candidates.sum() <= 10:
For each guess w (over full word list), compute the partition over candidates: for each resulting pattern bucket, the next-step optimal candidate count is approximated by bucket_size_after_optimal_guess ≈ 1 if bucket_size == 1, else the minimum over guesses of the max bucket size from the next split. Use the standard "expected guesses to finish" approximation E[guesses_remaining] = 1 + (bucket_size > 1) * expected_split_size / bucket_size summed over buckets.
Pick argmin E[total_guesses]. This prevents the BATCH/CATCH/HATCH trap.
Cap depth at 2 ply to keep cost bounded.
3. wordle/players.py — ModelPlayer
class ModelPlayer(Player):
    name = "Model"

    def __init__(self) -> None:
        from wordle.words import load_words
        from wordle.entropy_solver import EntropySolver
        print("Building/loading pattern matrix (~1 min on first run)...", flush=True)
        self._solver = EntropySolver(load_words())
        self._last_handled_history_len = -1

    def handle_event(self, event, game): return None

    def tick(self, game):
        if game.is_over:
            return None
        n = len(game.history)
        if n < self._last_handled_history_len:    # new round (R pressed)
            self._solver.reset()
            self._last_handled_history_len = -1
        if n > self._last_handled_history_len:
            guess = self._solver.choose_guess(game.history)
            self._last_handled_history_len = n
            return guess
        return None
Note: pattern-matrix build happens eagerly in __init__ (a few seconds when cached, ~1 min on first run). The UI briefly blocks during this — acceptable per the prompt and matches the user's expectation, much better than mid-game freeze. Per-guess choose_guess should be < 1 s after opener, so frame freezes are tolerable; threading is not added (keep simple, can revisit if visibly janky).

4. tests/test_entropy.py
Using pytest:

test_repeated_letters — BOBBY vs ABBEY → expected pattern (encoded). ALLEY vs LLAMA. SPEED vs ERASE. Cross-check against wordle.game.score_guess as oracle.
test_pattern_matrix_matches_oracle — build a small (50-word) pattern matrix, compare every entry to pattern_int(g, a).
test_filter_keeps_answer — pick a random answer, simulate 6 random guesses, after each choose_guess-induced filter the answer's index must remain set in candidates_mask.
test_entropy_reference — on a hand-picked 20-word list, compute entropy for each guess manually (or against a fresh independent implementation) and assert the top match.
test_solver_smoke — seed(42), sample 100 random answers, assert ≥ 95 solved within 6 guesses. Marked slow; can use pytest -k "not smoke" to skip.
5. scripts/run_entropy_simulation.py
CLI: --sample N (default: all words), --seed S (default: 0), --out PATH (default: results/entropy_run_<timestamp>.json).

Per-answer record: {answer, num_guesses, solved, turns: [{guess, pattern: [0..2]*5, entropy, candidates_remaining}]}.

Aggregate to stdout: mean, median, win-rate-within-6, worst case, histogram {1: c, 2: c, ..., 6: c, "fail": c}.

Single solver instance reused across all games (just call reset() between). Target: full 10.6k run in < 15 min on a laptop.

6. requirements.txt
Append:

numpy>=1.26
pytest>=7
7. README.md
Add a short subsection under "Plugging in a model":

### Entropy solver

The model toggle is wired to an information-theoretic entropy solver.

- First run builds an N×N pattern matrix (~1 min, ~113 MB) and an opener
  cache; subsequent runs load from `wordle/entropy_solver/cache/` in under
  5 s. Delete the cache folder to force a rebuild.
- Run a full benchmark: `python scripts/run_entropy_simulation.py --sample 500 --seed 42`
- Tests: `pytest tests/`
Performance Targets (Recalibrated)
The pool here is ~10.6k words (every word can be the answer), so it's slightly easier than the 13k figure in the original chat prompt. Targets:

Mean guesses on full pool: ≤ 4.0
Win rate within 6: ≥ 99%
First-run matrix build: < 90 s
Cached cold start: < 5 s
Per-guess decision after opener: < 1 s (likely < 0.5 s)
Full 10.6k simulation: < 15 min
Verification Plan
pip install -r requirements.txt
pytest tests/ — all pass; smoke test confirms ≥ 95/100 on seed 42.
python main.py --player model — UI launches; pattern matrix builds with progress message; UI runs; the Model plays an opener, then good follow-ups; R resets cleanly; multiple games in a row work.
python main.py then click "Player: Human / Model" mid-session — toggle works without crash; toast no longer says "not wired yet".
python scripts/run_entropy_simulation.py --sample 200 --seed 42 — runs to completion, prints aggregate stats, writes JSON under results/.
Inspect wordle/entropy_solver/cache/opener.json — contains top-10 openers with entropies (for report).
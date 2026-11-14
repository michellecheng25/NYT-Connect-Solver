"""
Hypothesis 3: does a player that updates its probabilities after each feedback
(correct / one away / wrong) do better than one that ignores the feedback?

    python evaluate_h3.py --sweep                      # pick the temperature on train puzzles
    python evaluate_h3.py                              # plain Numberbatch, all puzzles
    python evaluate_h3.py --embedding glove --split test

Both players use the same probabilities over all 2,627,625 full boards and guess the
most confident group first. Real game rules: a guess is correct, one away, or wrong,
and the game ends after 4 correct groups or 4 mistakes.
"""
import argparse
import csv
import os
import numpy as np
from src.data import load_puzzles, date_split
from src.embeddings import get_embedding
from src.scorer import plain_similarity
from src.solvers import play_boards, play
from evaluate_h2 import bootstrap_ci, paired_test, fmt

COLORS = ["yellow", "green", "blue", "purple"]
METRICS = ["groups_found", "won", "mistakes"]
PLAYERS = [("greedy", "greedy (ref)"), ("static", "no feedback"), ("feedback", "feedback")]
TEMPERATURES = [0.01, 0.02, 0.05, 0.1, 0.2]


def run(scores, truth, player, temperature):
    if player == "greedy":
        r = play(scores, truth)
    else:
        r = play_boards(scores, truth, temperature, use_feedback=(player == "feedback"))
    r["won"] = int(r["won"])
    r["found"] = [any(g == guess and fb == "correct" for guess, fb in r["history"]) for g in truth]
    return r


def sweep(puzzles, wv):
    """groups found on the train puzzles for each temperature"""
    scores = [plain_similarity(p["words"], wv) for p in puzzles]
    print(f"temperature sweep on {len(puzzles)} train puzzles (mean groups found / won)")
    for t in TEMPERATURES:
        line = f"  T={t:<5}"
        for player in ["static", "feedback"]:
            rs = [run(s, p["groups"], player, t) for s, p in zip(scores, puzzles)]
            line += f"  {player} {np.mean([r['groups_found'] for r in rs]):.3f} / {np.mean([r['won'] for r in rs]):.3f}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="numberbatch")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    splits = dict(zip(["train", "val", "test"], date_split(load_puzzles())))
    wv = get_embedding(args.embedding)
    if args.sweep:
        sweep(splits["train"], wv)
        return

    names = list(splits) if args.split == "all" else [args.split]
    puzzles = [p for s in names for p in splits[s]]
    split_of = [s for s in names for _ in splits[s]]

    results = {s: [] for s, _ in PLAYERS}
    for p in puzzles:
        scores = plain_similarity(p["words"], wv)
        for s, _ in PLAYERS:
            results[s].append(run(scores, p["groups"], s, args.temperature))

    n = len(puzzles)
    arr = {s: {m: np.array([r[m] for r in rs], dtype=float) for m in METRICS} for s, rs in results.items()}
    print(f"{args.embedding}, plain similarity, T={args.temperature}, {n} puzzles ({args.split}: "
          f"{puzzles[0]['date']} to {puzzles[-1]['date']})\n")

    print("=== 1. mean [95% bootstrap CI] ===")
    for s, name in PLAYERS:
        print(f"  {name:<13} " + "  |  ".join(f"{m} {fmt(*bootstrap_ci(arr[s][m]))}" for m in METRICS))

    print("\n=== 2. paired test: feedback - no feedback, same puzzles ===")
    for m in METRICS:
        mean, lo, hi, pb, pw = paired_test(arr["static"][m], arr["feedback"][m])
        print(f"  {m:<13} {fmt(mean, lo, hi, '+')}  p_boot {pb:.4f}  p_wilcoxon {pw:.2g}")
    d = arr["feedback"]["groups_found"] - arr["static"]["groups_found"]
    print(f"  groups_found: feedback better on {(d > 0).sum()}, worse on {(d < 0).sum()}, same on {(d == 0).sum()} puzzles")
    a, b = arr["static"]["won"].astype(bool), arr["feedback"]["won"].astype(bool)
    print(f"  won by both {(a & b).sum()}, feedback only {(b & ~a).sum()}, "
          f"no-feedback only {(a & ~b).sum()}, neither {(~a & ~b).sum()}")

    print("\n=== 3. % of real groups found, by color ===")
    for lv in range(4):
        pct = {s: np.mean([r["found"][p["levels"].index(lv)] for r, p in zip(results[s], puzzles)]) * 100
               for s, _ in PLAYERS}
        print(f"  {COLORS[lv]:<7} greedy {pct['greedy']:5.1f}%  no feedback {pct['static']:5.1f}%  "
              f"feedback {pct['feedback']:5.1f}%  ({pct['feedback'] - pct['static']:+.1f})")

    print("\n=== 4. how the games go ===")
    for s, name in PLAYERS:
        h = [fb for r in results[s] for _, fb in r["history"]]
        first = np.mean([r["history"][0][1] == "correct" for r in results[s]]) * 100
        won = arr[s]["won"].astype(bool)
        print(f"  {name:<13} guesses/game {len(h) / n:.2f}  correct {h.count('correct') / len(h) * 100:4.1f}%  "
              f"one away {h.count('one_away') / len(h) * 100:4.1f}%  wrong {h.count('wrong') / len(h) * 100:4.1f}%  |  "
              f"1st guess right {first:.1f}%  |  mistakes in won games {arr[s]['mistakes'][won].mean():.2f}")
    left = np.array([r["boards_left"] for r in results["feedback"]])
    print(f"  feedback player, boards still possible when the game ends: median {np.median(left):,.0f}, "
          f"exactly 1 in {(left == 1).mean() * 100:.1f}% of games")

    if len(names) > 1:
        print("\n=== 5. by split (groups found: no feedback -> feedback, difference [95% CI], Wilcoxon p) ===")
        for s in names:
            mask = np.array(split_of) == s
            mean, lo, hi, _, pw = paired_test(arr["static"]["groups_found"][mask], arr["feedback"]["groups_found"][mask])
            print(f"  {s:<6} n={mask.sum():<4} {arr['static']['groups_found'][mask].mean():.2f} -> "
                  f"{arr['feedback']['groups_found'][mask].mean():.2f}  ({fmt(mean, lo, hi, '+')})  p {pw:.2g}  |  "
                  f"won {arr['static']['won'][mask].mean() * 100:.0f}% -> {arr['feedback']['won'][mask].mean() * 100:.0f}%")

    #per-puzzle results, for plots and error analysis
    os.makedirs("results", exist_ok=True)
    path = f"results/h3_{args.embedding}_{args.split}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "split"] + [f"{s}_{m}" for s, _ in PLAYERS for m in METRICS])
        for i, p in enumerate(puzzles):
            w.writerow([p["date"], split_of[i]] + [results[s][i][m] for s, _ in PLAYERS for m in METRICS])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

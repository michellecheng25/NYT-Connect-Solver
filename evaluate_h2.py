"""
Hypothesis 2: does splitting all 16 words at once (whole-board) find more
correct groups than picking one group at a time (greedy)?

    python evaluate_h2.py                              # plain Numberbatch, all puzzles
    python evaluate_h2.py --embedding glove --split test

Both solvers get the same 1,820 plain-similarity scores and return 4 groups.
Nothing is trained or tuned, so every puzzle can be used, including the training ones.
"""
import argparse
import csv
import itertools
import os
import numpy as np
from scipy.stats import wilcoxon
from src.data import load_puzzles, date_split
from src.embeddings import get_embedding
from src.scorer import plain_similarity, COMBINATIONS
from src.solvers import whole_board, greedy_board, all_partitions

COLORS = ["yellow", "green", "blue", "purple"]
METRICS = ["groups_found", "solved", "words_correct"]
SOLVERS = [("random", "random board"), ("greedy", "greedy"), ("whole", "whole-board")]
GROUP_INDEX = {g: k for k, g in enumerate(COMBINATIONS)}
B = 10000  # bootstrap resamples
rng = np.random.default_rng(0)


def grade(pred, truth):
    """
    pred, truth: lists of 4 frozensets of board positions.
        groups_found   predicted groups that exactly match a real group (0-4)
        solved         1 if all 4 match
        words_correct  words in the right place (0-16), matching each predicted group
                       to a different real group so the overlap is as large as possible
        found          which real groups were found (for the color breakdown)
    """
    found = [g in pred for g in truth]
    words = max(sum(len(p & t) for p, t in zip(pred, perm)) for perm in itertools.permutations(truth))
    return {"groups_found": sum(found), "solved": int(all(found)), "words_correct": words, "found": found}


def random_board():
    """a random valid split of the 16 words, for reference"""
    P = all_partitions()
    return [frozenset(COMBINATIONS[k]) for k in P[rng.integers(len(P))]]


def true_board_rank(scores, truth):
    """position of the real answer among all 2,627,625 boards, by total score (1 = best)"""
    scores = np.asarray(scores, dtype=np.float64)
    totals = scores[all_partitions()].sum(axis=1)
    true_total = sum(scores[GROUP_INDEX[tuple(sorted(g))]] for g in truth)
    return 1 + int((totals > true_total + 1e-9).sum())


def bootstrap_ci(x):
    """mean and 95% confidence interval, resampling puzzles with replacement"""
    idx = rng.integers(0, len(x), (B, len(x)))
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_test(a, b):
    """b - a on the same puzzles: mean difference, 95% CI, bootstrap p, Wilcoxon p"""
    d = b - a
    mean, lo, hi = bootstrap_ci(d)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    p_boot = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    p_wil = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return mean, lo, hi, p_boot, p_wil


def fmt(mean, lo, hi, sign=""):
    return f"{mean:{sign}.2f} [{lo:{sign}.2f}, {hi:{sign}.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="numberbatch")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    splits = dict(zip(["train", "val", "test"], date_split(load_puzzles())))
    names = list(splits) if args.split == "all" else [args.split]
    puzzles = [p for s in names for p in splits[s]]
    split_of = [s for s in names for _ in splits[s]]
    wv = get_embedding(args.embedding)

    results = {s: [] for s, _ in SOLVERS}
    ranks = []
    for p in puzzles:
        scores = plain_similarity(p["words"], wv)
        results["random"].append(grade(random_board(), p["groups"]))
        results["greedy"].append(grade(greedy_board(scores), p["groups"]))
        results["whole"].append(grade(whole_board(scores), p["groups"]))
        ranks.append(true_board_rank(scores, p["groups"]))

    n = len(puzzles)
    arr = {s: {m: np.array([r[m] for r in rs], dtype=float) for m in METRICS} for s, rs in results.items()}
    ranks = np.array(ranks)
    print(f"{args.embedding}, plain similarity, {n} puzzles ({args.split}: "
          f"{puzzles[0]['date']} to {puzzles[-1]['date']})\n")

    print("=== 1. mean [95% bootstrap CI] ===")
    for s, name in SOLVERS:
        print(f"  {name:<13} " + "  |  ".join(f"{m} {fmt(*bootstrap_ci(arr[s][m]))}" for m in METRICS))

    print("\n=== 2. paired test: whole-board - greedy, same puzzles ===")
    for m in METRICS:
        mean, lo, hi, pb, pw = paired_test(arr["greedy"][m], arr["whole"][m])
        print(f"  {m:<14} {fmt(mean, lo, hi, '+')}  p_boot {pb:.4f}  p_wilcoxon {pw:.2g}")
    for m in ["groups_found", "words_correct"]:
        d = arr["whole"][m] - arr["greedy"][m]
        print(f"  {m}: whole-board better on {(d > 0).sum()}, worse on {(d < 0).sum()}, "
              f"same on {(d == 0).sum()} puzzles")
    g, w = arr["greedy"]["solved"].astype(bool), arr["whole"]["solved"].astype(bool)
    print(f"  solved by both {(g & w).sum()}, whole-board only {(w & ~g).sum()}, "
          f"greedy only {(g & ~w).sum()}, neither {(~g & ~w).sum()}")

    print("\n=== 3. % of real groups found, by color ===")
    for lv in range(4):
        pct = {s: np.mean([r["found"][p["levels"].index(lv)] for r, p in zip(results[s], puzzles)]) * 100
               for s, _ in SOLVERS}
        print(f"  {COLORS[lv]:<7} random {pct['random']:5.1f}%  greedy {pct['greedy']:5.1f}%  "
              f"whole-board {pct['whole']:5.1f}%  ({pct['whole'] - pct['greedy']:+.1f})")

    print("\n=== 4. how high does the real answer rank among all 2,627,625 boards? ===")
    print(f"  rank 1 (whole-board solves it): {(ranks == 1).mean() * 100:.1f}%")
    for k in [10, 100, 1000]:
        print(f"  in top {k:<5}: {(ranks <= k).mean() * 100:.1f}%")
    print(f"  median rank: {np.median(ranks):,.0f}")

    if len(names) > 1:
        print("\n=== 5. by split (groups found: greedy -> whole-board, difference [95% CI], Wilcoxon p) ===")
        for s in names:
            mask = np.array(split_of) == s
            mean, lo, hi, _, pw = paired_test(arr["greedy"]["groups_found"][mask], arr["whole"]["groups_found"][mask])
            print(f"  {s:<6} n={mask.sum():<4} {arr['greedy']['groups_found'][mask].mean():.2f} -> "
                  f"{arr['whole']['groups_found'][mask].mean():.2f}  ({fmt(mean, lo, hi, '+')})  p {pw:.2g}  |  "
                  f"solved {arr['greedy']['solved'][mask].mean() * 100:.0f}% -> {arr['whole']['solved'][mask].mean() * 100:.0f}%")

    #per-puzzle results, for plots and error analysis
    os.makedirs("results", exist_ok=True)
    path = f"results/h2_{args.embedding}_{args.split}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "split", "true_board_rank"] + [f"{s}_{m}" for s, _ in SOLVERS for m in METRICS])
        for i, p in enumerate(puzzles):
            w.writerow([p["date"], split_of[i], ranks[i]] + [results[s][i][m] for s, _ in SOLVERS for m in METRICS])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

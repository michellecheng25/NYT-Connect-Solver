"""
Hypothesis 3, guessing orders: which group should the feedback player guess next?

    python evaluate_h3_orders.py                       # plain Numberbatch, all puzzles
    python evaluate_h3_orders.py --split test

All three players use feedback and the same board probabilities; only the next guess differs.
    confident         the group with the highest probability, added up over all boards (H3 player)
    top_board         the highest-scoring group in the most likely board that's left
    top_board_random  a random unsolved group from the most likely board that's left
"""
import argparse
import csv
import os
import numpy as np
from src.data import load_puzzles, date_split
from src.embeddings import get_embedding
from src.scorer import plain_similarity
from src.solvers import play_boards, board_totals
from evaluate_h2 import bootstrap_ci, paired_test, fmt

COLORS = ["yellow", "green", "blue", "purple"]
METRICS = ["groups_found", "won", "mistakes"]
ORDERS = [("confident", "confident"), ("top_board", "top board, best"), ("top_board_random", "top board, random")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="numberbatch")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    splits = dict(zip(["train", "val", "test"], date_split(load_puzzles())))
    names = list(splits) if args.split == "all" else [args.split]
    puzzles = [p for s in names for p in splits[s]]
    split_of = [s for s in names for _ in splits[s]]
    wv = get_embedding(args.embedding)

    results = {o: [] for o, _ in ORDERS}
    for i, p in enumerate(puzzles):
        scores = plain_similarity(p["words"], wv)
        # shared by all three players
        totals = board_totals(scores)
        for o, _ in ORDERS:
            # seed = puzzle number, so the random order is the same every run
            r = play_boards(scores, p["groups"], args.temperature, True, totals, order=o, seed=i)
            r["won"] = int(r["won"])
            r["found"] = [any(g == guess and fb == "correct" for guess, fb in r["history"]) for g in p["groups"]]
            results[o].append(r)

    n = len(puzzles)
    arr = {o: {m: np.array([r[m] for r in rs], dtype=float) for m in METRICS} for o, rs in results.items()}
    print(f"{args.embedding}, plain similarity, feedback, T={args.temperature}, {n} puzzles ({args.split}: "
          f"{puzzles[0]['date']} to {puzzles[-1]['date']})\n")

    print("=== 1. mean [95% bootstrap CI] ===")
    for o, name in ORDERS:
        print(f"  {name:<18} " + "  |  ".join(f"{m} {fmt(*bootstrap_ci(arr[o][m]))}" for m in METRICS))

    print("\n=== 2. paired test: each order - confident, same puzzles ===")
    for o, name in ORDERS[1:]:
        print(f"  {name}")
        for m in METRICS:
            mean, lo, hi, pb, pw = paired_test(arr["confident"][m], arr[o][m])
            print(f"    {m:<13} {fmt(mean, lo, hi, '+')}  p_boot {pb:.4f}  p_wilcoxon {pw:.2g}")
        d = arr[o]["groups_found"] - arr["confident"]["groups_found"]
        print(f"    groups_found: {name} better on {(d > 0).sum()}, worse on {(d < 0).sum()}, same on {(d == 0).sum()} puzzles")

    print("\n=== 3. % of real groups found, by color ===")
    for lv in range(4):
        pct = {o: np.mean([r["found"][p["levels"].index(lv)] for r, p in zip(results[o], puzzles)]) * 100
               for o, _ in ORDERS}
        print(f"  {COLORS[lv]:<7} " + "  ".join(f"{name} {pct[o]:5.1f}%" for o, name in ORDERS))

    print("\n=== 4. how the games go ===")
    for o, name in ORDERS:
        h = [fb for r in results[o] for _, fb in r["history"]]
        first = np.mean([r["history"][0][1] == "correct" for r in results[o]]) * 100
        won = arr[o]["won"].astype(bool)
        in_won = f"{arr[o]['mistakes'][won].mean():.2f}" if won.any() else "n/a (no wins)"
        print(f"  {name:<18} guesses/game {len(h) / n:.2f}  correct {h.count('correct') / len(h) * 100:4.1f}%  "
              f"one away {h.count('one_away') / len(h) * 100:4.1f}%  wrong {h.count('wrong') / len(h) * 100:4.1f}%  |  "
              f"1st guess right {first:.1f}%  |  mistakes in won games {in_won}")

    if len(names) > 1:
        print("\n=== 5. by split (groups found, difference vs. confident [95% CI], Wilcoxon p) ===")
        for s in names:
            mask = np.array(split_of) == s
            line = f"  {s:<6} n={mask.sum():<4} confident {arr['confident']['groups_found'][mask].mean():.2f}"
            for o, name in ORDERS[1:]:
                mean, lo, hi, _, pw = paired_test(arr["confident"]["groups_found"][mask], arr[o]["groups_found"][mask])
                line += f"  |  {name} {fmt(mean, lo, hi, '+')} p {pw:.2g}"
            print(line)

    #per-puzzle results, for plots and error analysis
    os.makedirs("results", exist_ok=True)
    path = f"results/h3_orders_{args.embedding}_{args.split}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "split"] + [f"{o}_{m}" for o, _ in ORDERS for m in METRICS])
        for i, p in enumerate(puzzles):
            w.writerow([p["date"], split_of[i]] + [results[o][i][m] for o, _ in ORDERS for m in METRICS])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

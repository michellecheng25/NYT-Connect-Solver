"""
Play today's NYT Connections with the feedback player as your helper.

    python play_live.py
    python play_live.py --words "KNEE, ANKLE, SHIN, ..."     # skip typing the words

1. type the 16 words from the NYT site
2. the script suggests a guess; enter it on the NYT site (or pick another one)
3. type what NYT said: c = correct, o = one away, w = wrong
The script throws out every board that disagrees with the answer and suggests the next guess.
"""
import argparse
import numpy as np
from src.embeddings import get_embedding
from src.scorer import plain_similarity, COMBINATIONS
from src.solvers import BoardPlayer, GROUP_INDEX

FEEDBACK = {"c": "correct", "o": "one_away", "w": "wrong"}


def ask_words():
    while True:
        words = [w.strip().upper() for w in input("16 words, separated by commas: ").split(",") if w.strip()]
        if len(words) == 16 and len(set(words)) == 16:
            return words
        print(f"  got {len(words)} words ({len(set(words))} different), need 16 different words")


def ask_guess(words, options, player):
    """which group was entered on the NYT site: a suggestion number, or 4 words"""
    n = len(options)
    while True:
        ans = input(f"Which did you guess? [1] (1-{n}, or type 4 words): ").strip()
        if ans == "" or (ans.isdigit() and 1 <= int(ans) <= n):
            return options[int(ans or 1) - 1][0]
        picked = [w.strip().upper() for w in ans.split(",")]
        if len(picked) == 4 and len(set(picked)) == 4 and all(w in words for w in picked):
            guess = frozenset(words.index(w) for w in picked)
            if GROUP_INDEX[tuple(sorted(guess))] in player.guessed:
                print("  you already guessed that one; NYT doesn't count repeats, so enter a different guess")
                continue
            if not any(guess & g for g in player.solved):
                return guess
        print(f"  need 1-{n}, or 4 different words from the board (not already solved), separated by commas")


def ask_feedback():
    while True:
        ans = input("What did NYT say? (c = correct, o = one away, w = wrong): ").strip().lower()
        if ans in FEEDBACK:
            return FEEDBACK[ans]


def show(group, words):
    return ", ".join(words[i] for i in sorted(group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="numberbatch")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--words", help="the 16 words, separated by commas")
    args = parser.parse_args()

    print(f"loading {args.embedding}...")
    wv = get_embedding(args.embedding)
    words = [w.strip().upper() for w in args.words.split(",")] if args.words else ask_words()
    if len(words) != 16 or len(set(words)) != 16:
        raise SystemExit("--words needs 16 different words")

    unknown = [w for w in words if not np.any(wv.vec(w))]
    if unknown:
        print(f"  not in {args.embedding} (the solver knows nothing about these): {', '.join(unknown)}")

    player = BoardPlayer(plain_similarity(words, wv), args.temperature, use_feedback=True)
    while not player.done():
        left = [w for i, w in enumerate(words) if not any(i in g for g in player.solved)]
        print(f"\nmistakes left: {4 - player.mistakes}  |  boards still possible: {len(player.boards):,}")
        print(f"words left: {', '.join(left)}")
        options = player.top_guesses(3)
        for n, (g, p) in enumerate(options, 1):
            print(f"  {n}. {show(g, words):<40} {p * 100:5.1f}% sure")

        guess = ask_guess(words, options, player)
        player.give_feedback(guess, ask_feedback())
        if len(player.boards) == 0:
            raise SystemExit("\nno board fits all the answers so far. was one typed in wrong?")

    if len(player.solved) == 4:
        print(f"\nsolved with {player.mistakes} mistake(s)!")
    else:
        print("\nout of mistakes. the most likely board that's left:")
        for k in player.boards[np.argmax(player.logp)]:
            g = frozenset(COMBINATIONS[k])
            if g not in player.solved:
                print(f"  {show(g, words)}")


if __name__ == "__main__":
    main()

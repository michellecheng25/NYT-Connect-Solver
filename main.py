from src.data import load_puzzles;
from src.embeddings import WordVectors;
from src.scorer import plain_similarity;
from src.search import greedy_search
from src.evaluate import count_correct_groups

import numpy as np



'''def main():
    wv = WordVectors.load_glove()
    words = ['SNOW', 'LEVEL', 'SHIFT', 'KAYAK', 'HEAT', 'TAB', 'BUCKS', 'RETURN', 'JAZZ', 'HAIL', 'OPTION', 'RAIN', 'SLEET', 'RACECAR', 'MOM', 'NETS']
    scores = plain_similarity(words, wv)
    print(scores[:10])
    print(len(scores))'''



def main():

    print("Loading GloVe...")
    wv = WordVectors.load_glove()

    vector = wv.vec("SNOW")
    print("SNOW vector shape:", vector.shape)
    print("First five numbers:", vector[:5])

    puzzles = load_puzzles()

    if not puzzles:
        raise ValueError("No complete puzzles were loaded.")

    puzzle = puzzles[0]

    scores = plain_similarity(puzzle["words"], wv)

    predicted_groups = greedy_search(scores)

    correct = count_correct_groups(predicted_groups, puzzle["groups"])
    print(f"\nCorrect groups: {correct}/4")

    print("\nPredicted groups:")
    for number, group in enumerate(predicted_groups, start=1):
        words = [puzzle["words"][i] for i in sorted(group)]
        print(f"{number}: {', '.join(words)}")

    print(f"\nCandidates scored: {len(scores)}")
    print("First five scores:", scores[:5])

    print(f"Loaded {len(puzzles)} puzzles.")
    print(f"\nPuzzle {puzzle['id']} — {puzzle['date']}")

    print("\nWords:")
    for row in range(4):
        start = row * 4
        print(" | ".join(puzzle["words"][start:start + 4]))

    print("\nCorrect groups:")
    for name, indices in zip(puzzle["group_names"], puzzle["groups"]):
        words = [puzzle["words"][i] for i in sorted(indices)]
        print(f"{name}: {', '.join(words)}")


    demo_puzzles = puzzles[:10]
    total_correct = 0
    fully_solved = 0

    print("\nResults across 10 puzzles:")

    for demo in demo_puzzles:
        demo_scores = plain_similarity(demo["words"], wv)
        predictions = greedy_search(demo_scores)
        correct = count_correct_groups(predictions, demo["groups"])

        total_correct += correct
        if correct == 4:
            fully_solved += 1

        print(f"{demo['date']}: {correct}/4 correct groups")

    count = len(demo_puzzles)
    print(f"\nPuzzles evaluated: {count}")
    print(f"Average correct groups: {total_correct / count:.2f}/4")
    print(f"Fully solved: {fully_solved / count:.1%}")



if __name__ == "__main__":
    main()
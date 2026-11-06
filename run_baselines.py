import random

from src.baselines import kmeans_grouping, random_grouping
from src.data import date_split, load_puzzles
from src.evaluate import count_correct_groups
from src.embeddings import get_embedding


# random baseline
# repeats the baseline algorithm to get a larger sample size for evaluation
REPEATS = 1000

rng = random.Random(42)
_, _, test = date_split(load_puzzles())

total_found = 0
total_solved = 0
for puzzle in test:
    for _ in range(REPEATS):
        found = count_correct_groups(random_grouping(rng), puzzle["groups"])
        total_found += found
        total_solved += found == 4

trials = len(test) * REPEATS
print(f"Random baseline over {trials} trials:")
print(f"Test puzzles: {len(test)}")
print(f"Average groups found: {total_found / trials:.4f} / 4")
print(f"Fully solved: {total_solved}/{trials}")

# k-means baseline with glove embeddings
embedding = get_embedding("glove")
kmeans_rng = random.Random(42)

found = [
    count_correct_groups(
        kmeans_grouping(embedding.vectors(p["words"]), kmeans_rng), p["groups"]
    )
    for p in test
]
print(f"K-means + GloVe baseline over {len(test)} test puzzles:")
print(f"Average groups found: {sum(found) / len(found):.3f} / 4 groups")
print(f"Fully solved: {sum(f == 4 for f in found)}/{len(found)}")
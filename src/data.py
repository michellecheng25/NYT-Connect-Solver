
from datasets import load_dataset

def load_puzzles():
    """Return training puzzles sorted by date, oldest first.

    Each puzzle is a dict:
        id      int
        date    "YYYY-MM-DD"
        words   list of 16 strings, in board order
        groups  list of 4 sets of word indices (true groups)
        levels (difficulty)  list of 4 ints, based on difficulty of groups
    """
    
    df = load_dataset("eric27n/NYT-Connections")["train"].to_pandas()
    puzzles = []

    #group rows together into a game by game ID
    for gid, g in df.groupby("Game ID"):
        #set idx from 0 - 15 for demo puproses
        g = g.reset_index(drop=True)

        #skip malformed puzzles (# words != 16, # of unique groups != 4)
        if len(g) != 16 or g["Group Name"].nunique() != 4:
            continue

        # clean words
        words = [str(w).strip() for w in g["Word"]]

        group_names, groups, levels = [], [], []
        #for each group in apuzzle, get the words and level
        for name, row in g.groupby("Group Name"):
            group_names.append(name)
            groups.append(frozenset(row.index.tolist()))
            levels.append(int(row["Group Level"].iloc[0]))

        puzzles.append({
            "id": int(gid),
            "date": str(g["Puzzle Date"].iloc[0]),
            "group_names": group_names,
            "words": words,
            "groups": groups,
            "levels": levels,
        })
    puzzles.sort(key=lambda p: p["date"])
    return puzzles

#split puzzle for train, val, and test
def date_split(puzzles, train=0.70, val=0.15):
    n = len(puzzles)
    a, b = int(n * train), int(n * (train + val))
    return puzzles[:a], puzzles[a:b], puzzles[b:]


def view_puzzles():
    for puzzle in load_puzzles()[:1]:
        print(puzzle["date"])
        for i, group in enumerate(puzzle["groups"]):
            level = puzzle["levels"][i]
            print(f'{puzzle["group_names"][i]} (level {level}): ', end="")
            for idx in sorted(group):
                print(puzzle["words"][idx], end=" ")
            print()

#view_puzzles()


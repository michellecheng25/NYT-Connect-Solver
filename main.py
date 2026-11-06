from src.data import load_puzzles;
from src.embeddings import GloVe, Numberbatch, get_embedding;
from src.scorer import plain_similarity;
from src.solvers import GreedyPlayer, play;




def test():
    embedding = get_embedding("numberbatch")
    puzzle = ['SNOW', 'CARRY ON', 'SHIFT', 'CONTINUE', 'HEAT', 'TAB', 'BUCKS', 'RETURN', 'JAZZ', 'HAIL', 'OPTION', 'RAIN', 'SLEET', 'PROCEED', 'RESUME', 'NETS']
    scores = plain_similarity(puzzle, embedding)
    truth = [
        frozenset({0,9,11,12}), #wet weather
        frozenset({1,3,13,14}), #keep going
        frozenset({2,5,7,10}), #keyboard keys
        frozenset({4,6,8,15}) #nba teams
    ]
    result = play(scores, truth)
    print(result)


def main():
    embedding = get_embedding("numberbatch")
    puzzles = load_puzzles()[:10]

    total_found, total_solved = 0, 0
    for p in puzzles:
        scores = plain_similarity(p["words"], embedding)
        result = play(scores, p["groups"])
        total_found += result["groups_found"]
        total_solved += result["won"]
        print(f"{p['date']}  found {result['groups_found']}/4  "
              f"mistakes {result['mistakes']}  "
              f"{'SOLVED' if result['won'] else ''}")

    n = len(puzzles)
    print(f"\navg groups found: {total_found / n:.2f} / 4")
    print(f"fully solved: {total_solved}/{n}")
    print(f"words missing from embedding: {embedding.missing}/{embedding.total}")


main()
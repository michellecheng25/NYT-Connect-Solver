from src.data import load_puzzles;
from src.embeddings import GloVe, Numberbatch, get_embedding;
from src.scorer import plain_similarity;
from src.solvers import GreedyPlayer, play;




def main():
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
    


main()
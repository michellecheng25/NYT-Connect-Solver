import numpy as np
from src.scorer import COMBINATIONS

class GreedyPlayer:
    """
    A greedy player you can step through one turn at a time.
    The scores are sorted once, when the player is created.
    """

    def __init__(self, scores, use_one_away=True):
        # groups still possible to guess, best score first
        self.alive = [frozenset(COMBINATIONS[k]) for k in np.argsort(-scores)]
        self.use_one_away = use_one_away
        self.solved = []         # groups we got right
        self.mistakes = 0
        self.near_miss = None    # last "one away" guess
        self.history = []        # (guess, feedback) for each turn

    def done(self):
        return len(self.solved) == 4 or self.mistakes >= 4

    def next_guess(self):
        # after "one away", prefer the best group sharing exactly 3 words with it
        if self.use_one_away and self.near_miss is not None:
            for g in self.alive:
                if len(g & self.near_miss) == 3:
                    return g
        
        #return group with the highest score
        return self.alive[0]

    def give_feedback(self, guess, feedback):
        self.history.append((guess, feedback))
        self.alive.remove(guess)                      # never guess it again
        if feedback == "correct":
            self.solved.append(guess)
            # its words are off the board: drop every group that touches them
            self.alive = [g for g in self.alive if g.isdisjoint(guess)]
            self.near_miss = None
        else:
            self.mistakes += 1
            if feedback == "one_away":
                self.near_miss = guess


def get_feedback(guess, truth):
    """
    guess: frozenset of 4 board positions.
    truth: list of 4 frozensets (the real groups).
    returns "correct", "one_away" (3 of the 4 words belong together), or "wrong".
    """
    if guess in truth:
        return "correct"
    for group in truth:
        if len(guess & group) == 3:
            return "one_away"
    return "wrong"

def play(scores, truth, use_one_away=True):
    player = GreedyPlayer(scores, use_one_away)
    while not player.done():
        guess = player.next_guess()
        player.give_feedback(guess, get_feedback(guess, truth))

    return {"won": len(player.solved) == 4, "groups_found": len(player.solved),
            "mistakes": player.mistakes, "history": player.history}
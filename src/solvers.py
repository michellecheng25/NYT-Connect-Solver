import numpy as np
from src.scorer import COMBINATIONS

# each candidate group as a 16-bit mask of its board positions, e.g. {0,1,2,3} -> 0b1111
MASKS = np.array([sum(1 << i for i in g) for g in COMBINATIONS])
# lowest board position in each group (combinations are sorted, so it's the first one)
LOWEST = np.array([g[0] for g in COMBINATIONS])
# mask -> group index, to find the last group from the 4 words that are left
MASK_TO_GROUP = np.full(1 << 16, -1)
MASK_TO_GROUP[MASKS] = np.arange(len(COMBINATIONS))
FULL = (1 << 16) - 1

_partitions = None


def _lowest_unused(used):
    """board position of the lowest word not in the used mask"""
    return ((~used) & (used + 1)).bit_length() - 1


def all_partitions():
    """
    every way to split 16 words into 4 groups of 4: (2,627,625, 4) array of group indices.
    to avoid counting the same split twice, each next group must contain the lowest
    word not used yet: 455 * 165 * 35 * 1 = 2,627,625.
    built once (takes a couple of seconds), then reused.
    """
    global _partitions
    if _partitions is not None:
        return _partitions

    rows = []
    for a in np.flatnonzero(LOWEST == 0):
        used_a = int(MASKS[a])
        for b in np.flatnonzero((LOWEST == _lowest_unused(used_a)) & (MASKS & used_a == 0)):
            used_b = used_a | int(MASKS[b])
            c = np.flatnonzero((LOWEST == _lowest_unused(used_b)) & (MASKS & used_b == 0))
            d = MASK_TO_GROUP[FULL ^ used_b ^ MASKS[c]]
            rows.append(np.stack([np.full(len(c), a), np.full(len(c), b), c, d], axis=1))

    _partitions = np.concatenate(rows).astype(np.int16)
    return _partitions


def whole_board(scores):
    """
    split all 16 words at once: the 4 non-overlapping groups with the highest total score.
    returns a list of 4 frozensets of board positions.
    """
    P = all_partitions()
    #float64 so near-equal boards aren't decided by float32 rounding
    scores = np.asarray(scores, dtype=np.float64)
    best = P[np.argmax(scores[P].sum(axis=1))]
    return [frozenset(COMBINATIONS[k]) for k in best]


def greedy_board(scores):
    """
    split all 16 words one group at a time: take the best group, remove its words, repeat.
    returns a list of 4 frozensets of board positions.
    """
    picked, used = [], 0
    for k in np.argsort(-scores):
        if MASKS[k] & used == 0:
            picked.append(frozenset(COMBINATIONS[k]))
            used |= int(MASKS[k])
            if len(picked) == 4:
                break
    return picked

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
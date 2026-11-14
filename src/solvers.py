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
# group (sorted tuple of positions) -> index into COMBINATIONS
GROUP_INDEX = {g: k for k, g in enumerate(COMBINATIONS)}
# number of 1 bits in every 16-bit mask, so POPCOUNT[a & b] = words two groups share
POPCOUNT = np.array([bin(m).count("1") for m in range(1 << 16)], dtype=np.int8)

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


class BoardPlayer:
    """
    Keeps a probability for every full board (all 2,627,625 ways to split the 16 words).
        P(board) is proportional to exp(total score / temperature)
    Each turn it guesses the group with the highest probability, adding up over every
    board that contains it (the "most confident" group).

    use_feedback=True: after each guess, throw out the boards that disagree with the feedback
        correct   keep boards that contain the guess
        one away  keep boards with a group sharing exactly 3 words with the guess
        wrong     keep boards where no group shares 3 or more words with the guess
    use_feedback=False: probabilities are fixed at the start. It still can't reuse words
        from groups it already got right (the game removes them), and never repeats a guess.
    """

    def __init__(self, scores, temperature=0.05, use_feedback=True):
        self.boards = all_partitions()
        self.logp = np.asarray(scores, dtype=np.float64)[self.boards].sum(axis=1) / temperature
        self.use_feedback = use_feedback
        self.marginal = self._marginals()
        self.solved = []         # groups we got right
        self.used = 0            # mask of words in solved groups
        self.guessed = set()     # group indices already tried
        self.mistakes = 0
        self.history = []        # (guess, feedback) for each turn

    def _marginals(self):
        """probability of each of the 1,820 groups = sum over the boards that contain it"""
        p = np.exp(self.logp - self.logp.max())
        p /= p.sum()
        return np.bincount(self.boards.ravel(), weights=np.repeat(p, 4), minlength=len(COMBINATIONS))

    def done(self):
        return len(self.solved) == 4 or self.mistakes >= 4

    def next_guess(self):
        for k in np.argsort(-self.marginal):
            if k not in self.guessed and MASKS[k] & self.used == 0:
                return frozenset(COMBINATIONS[k])

    def give_feedback(self, guess, feedback):
        self.history.append((guess, feedback))
        k = GROUP_INDEX[tuple(sorted(guess))]
        self.guessed.add(k)
        if feedback == "correct":
            self.solved.append(guess)
            self.used |= int(MASKS[k])
        else:
            self.mistakes += 1

        if self.use_feedback:
            # most words the guess shares with any one group, on each board
            overlap = POPCOUNT[MASKS[k] & MASKS][self.boards].max(axis=1)
            keep = {"correct": overlap == 4, "one_away": overlap == 3, "wrong": overlap < 3}[feedback]
            self.boards, self.logp = self.boards[keep], self.logp[keep]
            self.marginal = self._marginals()



def play_boards(scores, truth, temperature=0.05, use_feedback=True):
    player = BoardPlayer(scores, temperature, use_feedback)
    while not player.done():
        guess = player.next_guess()
        player.give_feedback(guess, get_feedback(guess, truth))

    return {"won": len(player.solved) == 4, "groups_found": len(player.solved),
            "mistakes": player.mistakes, "history": player.history,
            "boards_left": len(player.boards)}
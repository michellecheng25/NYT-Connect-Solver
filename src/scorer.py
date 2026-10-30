
import itertools
import numpy as np

#16 words per puzzle, four groups a puzzle is 1820 combinations
combinations = list(itertools.combinations(range(16), 4))

# 4 words per group, comapre combinations of two words in a group (cosine similarity)
pairs = list(itertools.combinations(range(4), 2))


def plain_similarity(words, wv):
    """
    takes 16 words from a puzzle and returns a numpy array of 16C4 (1,820) scores
    representing scores of every combination of 4 groups that can be made from 16 words.
    """

    V = np.array([wv.vec(w) for w in words])

    #normalize all word vectors
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    #compares every word vector in candiate group to every other word in that group
    S = V @ V.T

    #gets the average of the similarity b/w all combinations of pair of words in the group
    scores = []
    for group in combinations:
        pair_sims = []
        for i, j in pairs:
            pair_sims.append(S[group[i], group[j]])
        scores.append(np.mean(pair_sims))
    
    return np.array(scores)




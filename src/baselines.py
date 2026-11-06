import numpy as np

def random_grouping(rng):
    """Randomly split board positions 0-15 into four groups of four."""
    positions = list(range(16))
    rng.shuffle(positions)
    return [frozenset(positions[i:i + 4]) for i in range(0, 16, 4)]

def _kmeans_once(X, rng, max_iters=50):
    centers = X[rng.sample(range(16), 4)]
    assignment = None

    for _ in range(max_iters):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assignment = distances.argmin(axis=1)

        if assignment is not None and np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment

        # A center that loses all its words stays where it is.
        centers = np.array([
            X[assignment == c].mean(axis=0) if (assignment == c).any() else centers[c]
            for c in range(4)
        ])

    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1)
    inertia = float(distances.min(axis=1).sum())
    return assignment, inertia

def kmeans_grouping(vectors, rng, n_starts=100):
    """Cluster 16 word vectors with regular k-means. Clusters can be any size."""
    X = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    best_assignment, best_inertia = None, np.inf

    for _ in range(n_starts):
        assignment, inertia = _kmeans_once(X, rng)
        if inertia < best_inertia:
            best_assignment, best_inertia = assignment, inertia

    return [
        frozenset(np.flatnonzero(best_assignment == c).tolist())
        for c in range(4)
        if (best_assignment == c).any()
    ]
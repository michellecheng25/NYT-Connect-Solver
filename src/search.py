from src.scorer import combinations


def greedy_search(scores):
    # Consider candidates from highest score to lowest.
    ranked = sorted(
        range(len(combinations)),
        key=lambda i: scores[i],
        reverse=True,
    )

    selected = []
    used_words = set()

    for i in ranked:
        group = frozenset(combinations[i])

        # Skip groups containing any already-selected words.
        if group & used_words:
            continue

        selected.append(group)
        used_words.update(group)

        if len(selected) == 4:
            break

    return selected
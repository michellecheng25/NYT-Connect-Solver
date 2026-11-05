def count_correct_groups(predicted_groups, true_groups):
    true_sets = {frozenset(group) for group in true_groups}
    predicted_sets = {frozenset(group) for group in predicted_groups}

    return len(predicted_sets & true_sets)
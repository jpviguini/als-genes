


# precision at k - fraction of relevant items in top k results
def calculate_precision_at_k(ranked_list, validation_set, k):
    top_k = ranked_list[:k]
    hits = len(set(top_k) & validation_set)

    return hits / k if k > 0 else 0

# recall at k - fraction of relevant items found in top k
def calculate_recall_at_k(ranked_list, validation_set, k):

    top_k = ranked_list[:k]
    hits = len(set(top_k) & validation_set)

    return hits / len(validation_set) if len(validation_set) > 0 else 0

# mean reciprocal rank of first relevant item
def calculate_mrr(ranked_list, validation_set):

    for i, item in enumerate(ranked_list):
        if item in validation_set:
            return 1 / (i + 1)
    return 0.0
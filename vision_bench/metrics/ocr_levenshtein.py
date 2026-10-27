def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(pred: str, target: str) -> float:
    """Character Error Rate (CER) normalized by target length."""
    if not target:
        return 0.0 if not pred else 1.0
    dist = levenshtein_distance(pred, target)
    return round(min(1.0, dist / len(target)), 4)

def normalized_levenshtein_similarity(pred: str, target: str) -> float:
    """Similarity score in range [0.0, 1.0]."""
    max_len = max(len(pred), len(target))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred, target)
    return round(1.0 - (dist / max_len), 4)

def table_cell_similarity(pred_cells, gt_cells):
    return 1.0 if pred_cells == gt_cells else 0.85

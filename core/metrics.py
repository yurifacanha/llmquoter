from rank_bm25 import BM25Okapi

from core.quote_utils import parse_quotes


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def f1_score(precision: float, recall: float) -> float:
    precision = max(0.0, min(1.0, precision))
    recall = max(0.0, min(1.0, recall))
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def bm25_score(ground_truth: str, system_response: str) -> float:
    gt_quotes = parse_quotes(ground_truth)
    sys_quotes = parse_quotes(system_response)
    if not gt_quotes or not sys_quotes:
        return 0.0
    gt_tokenized = [tokenize(q) for q in gt_quotes]
    bm25 = BM25Okapi(gt_tokenized)
    sys_score = 0.0
    for sys_quote in sys_quotes:
        scores = bm25.get_scores(tokenize(sys_quote))
        sys_score += max(scores) if len(scores) > 0 else 0
    perfect_score = 0.0
    for gt_quote in gt_quotes:
        scores = bm25.get_scores(tokenize(gt_quote))
        perfect_score += max(scores) if len(scores) > 0 else 0
    if perfect_score == 0:
        return 0.0
    normalized = sys_score / perfect_score
    return round(min(1.0, normalized), 4)

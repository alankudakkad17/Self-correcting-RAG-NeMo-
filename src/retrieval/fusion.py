"""Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum over rankings of 1 / (k + rank(d))."""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of doc ids (best first). Ranks start at 1.

    Ties are broken by first appearance (earliest list, then earliest rank), so the output is deterministic.
    A doc id repeated within one list only counts at its best rank.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen_here = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_here:
                continue
            seen_here.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))

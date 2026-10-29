"""Sparse retrieval with BM25 (rank_bm25). Rebuilt in memory at load time: ~3.6k passages take < 1 s."""

import re

from rank_bm25 import BM25Okapi

from src.state import Passage

# Small stopword list: enough to stop "the/of/and" dominating short multi-hop questions.
STOPWORDS = set("""a an and are as at be by did do does for from had has have he her his how in is it its
of on or she that the their they this to was were what when where which who whom whose why will with""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class BM25Index:

    def __init__(self, passages: list[Passage]):
        self.doc_ids = [p.doc_id for p in passages]
        # Title is part of the document: multi-hop questions often name the entity.
        self._bm25 = BM25Okapi([tokenize(f"{p.title} {p.text}") for p in passages])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]  # index breaks ties: deterministic
        return [(self.doc_ids[i], float(scores[i])) for i in top]

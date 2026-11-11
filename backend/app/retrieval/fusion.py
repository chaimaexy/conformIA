"""Reciprocal Rank Fusion (Cormack et al., 2009).

Chaque retriever (dense, BM25, reformulations multi-query...) produit un classement. Leurs
scores ne sont pas comparables (cosinus vs BM25), mais leurs RANGS le sont :
    RRF(d) = somme sur les classements r de 1 / (k + rang_r(d))
k = 60 (valeur de l'article) amortit l'écart entre les tout premiers rangs : un document bien
classé par plusieurs retrievers passe devant un document premier d'un seul classement.
"""


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

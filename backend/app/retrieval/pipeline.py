"""Pipeline de recherche : multi-query, retrievers denses (pgvector) + BM25, RRF, reranking.

    question ─> [multi-query : + 3 reformulations Mistral]
      pour chaque version ─┬─ dense mistral-embed (Supabase, HNSW) ─┐
                           ├─ dense bge-m3        (Supabase, HNSW) ─┼─ RRF ─> 20 candidats
                           └─ BM25 (mémoire, racines françaises)  ──┘
      ─> [cross-encodeur : rescore des 20 candidats] ─> top k (+ score pour le garde-fou)

Chaque étape est activable séparément pour être mesurée sur le golden set
(python -m app.evaluation.retrieval --suite phase4).
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

from supabase import Client

from app.core.config import get_app_config
from app.embeddings import get_embedder
from app.retrieval.bm25 import BM25
from app.retrieval.fusion import rrf
from app.vectordb.store import client, vector_literal

PAGE = 1000  # limite de lignes par requête PostgREST
_CHUNKS: dict[str, dict[str, dict]] = {}  # stratégie -> chunks, partagé entre configurations
_PART = re.compile(r"\(partie (\d+)/\d+")  # « Article 99 - Sanctions (partie 2/5 - ... »


def part_number(chunk: dict) -> int:
    m = _PART.search(chunk["metadata"]["header"])
    return int(m[1]) if m else 1


@dataclass(frozen=True)
class RetrievalSettings:
    strategy: str = "structural"
    dense_models: tuple[str, ...] = ("mistral-embed",)
    bm25: bool = False
    candidates: int = 50  # résultats demandés à chaque retriever avant fusion
    rrf_k: int = 60
    multi_query: bool = False
    reranker: str | None = None  # « cohere:<modèle> » ou id Hugging Face (voir rerank.py)
    rerank_candidates: int = 20  # candidats fusionnés relus par le cross-encodeur (CPU : coûteux)
    # section découpée : jointe en entier jusqu'à cette taille, sinon parties voisines (0 : non)
    parent_max_chars: int = 0

    @classmethod
    def from_config(cls) -> "RetrievalSettings":
        """Configuration de production (config.yaml : chunking, retrieval, reranker)."""
        cfg = get_app_config()
        r = cfg.retrieval
        return cls(
            strategy=cfg.chunking.strategy,
            dense_models=tuple(r.dense_models),
            bm25=r.bm25,
            candidates=r.candidates,
            rrf_k=r.rrf_k,
            multi_query=r.multi_query,
            reranker=cfg.reranker.model,
            rerank_candidates=r.rerank_candidates,
            parent_max_chars=r.parent_max_chars,
        )


def load_chunks(db: Client, strategy: str) -> list[dict]:
    rows: list[dict] = []
    while True:
        page = (
            db.table("chunks")
            .select("id, document_id, content, metadata")
            .eq("strategy", strategy)
            .order("id")
            .range(len(rows), len(rows) + PAGE - 1)
            .execute()
            .data
        )
        rows += page
        if len(page) < PAGE:
            return rows


class Retriever:
    def __init__(self, settings: RetrievalSettings, db: Client | None = None):
        self.settings = settings
        self.db = db or client()

    @property
    def chunks(self) -> dict[str, dict]:
        strategy = self.settings.strategy
        if strategy not in _CHUNKS:
            _CHUNKS[strategy] = {row["id"]: row for row in load_chunks(self.db, strategy)}
        return _CHUNKS[strategy]

    @cached_property
    def bm25(self) -> tuple[BM25, list[str]]:
        ids = list(self.chunks)
        return BM25([self.chunks[i]["content"] for i in ids]), ids

    def _dense(self, vector, model: str) -> list[dict]:
        return (
            self.db.rpc(
                "match_chunks_dense",
                {
                    "query_embedding": vector_literal(vector),
                    "match_count": self.settings.candidates,
                    "filter_strategy": self.settings.strategy,
                    "embedding_model": model,
                },
            )
            .execute()
            .data
        )

    def rankings(self, queries: list[str]) -> list[list[list[tuple[str, float]]]]:
        """Pour chaque requête, un classement [(chunk_id, score brut)] par retriever."""
        per_query: list[list[list[tuple[str, float]]]] = [[] for _ in queries]
        for model in self.settings.dense_models:
            vectors = get_embedder(model).embed(queries)  # un seul appel (lots + cache)
            for i, vector in enumerate(vectors):
                per_query[i].append([(r["id"], r["score"]) for r in self._dense(vector, model)])
        if self.settings.bm25:
            index, ids = self.bm25
            for i, query in enumerate(queries):
                hits = index.search(query, self.settings.candidates)
                per_query[i].append([(ids[j], score) for j, score in hits])
        return per_query

    @cached_property
    def reranker(self):
        from app.retrieval.rerank import get_reranker

        return get_reranker(self.settings.reranker)

    def search_many(self, questions: list[str], k: int = 20) -> list[list[dict]]:
        if self.settings.multi_query:
            from app.retrieval.multiquery import expand

            variants = [[q, *expand(q)] for q in questions]
        else:
            variants = [[q] for q in questions]
        flat = [v for vs in variants for v in vs]
        flat_rankings = self.rankings(flat)  # embeddings calculés en un lot pour tout
        results, pos = [], 0
        for question, vs in zip(questions, variants, strict=True):
            ranked = [r for rankings in flat_rankings[pos : pos + len(vs)] for r in rankings]
            pos += len(vs)
            if len(ranked) == 1:  # un seul retriever : on garde son score (cosinus ou BM25)
                fused = ranked[0]
            else:
                fused = rrf([[doc for doc, _ in r] for r in ranked], self.settings.rrf_k)
            if self.settings.reranker:
                fused = self._rerank(question, fused[: self.settings.rerank_candidates])
            results.append([self._row(doc, score) for doc, score in fused[:k]])
        return results

    def _rerank(self, question: str, fused: list[tuple[str, float]]) -> list[tuple[str, float]]:
        ids = [doc for doc, _ in fused]
        scores = self.reranker.scores(question, [self._row(i, 0)["content"] for i in ids])
        return sorted(zip(ids, scores, strict=True), key=lambda kv: kv[1], reverse=True)

    @cached_property
    def section_parts(self) -> dict[tuple[str, str], list[dict]]:
        """(document, « Article 99 ») -> chunks de la section, dans l'ordre du texte."""
        parts: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for chunk in self.chunks.values():
            parts[(chunk["document_id"], chunk["metadata"]["label"])].append(chunk)
        for chunks in parts.values():
            chunks.sort(key=part_number)
        return dict(parts)

    @cached_property
    def section_heads(self) -> dict[tuple[str, str], dict]:
        """(document, « Article 15 ») -> 1er chunk de la section (pour l'expansion)."""
        return {key: chunks[0] for key, chunks in self.section_parts.items()}

    def with_siblings(self, rows: list[dict], scored: dict[str, float]) -> list[dict]:
        """Complète chaque chunk d'une section découpée par ses autres parties (parent document).

        Le chunking structurel coupe un long article en parties d'au plus 2 000 caractères :
        la partie retrouvée (article 99, paragraphes 4 à 6 : plafond des PME) peut renvoyer à une
        autre (paragraphe 3 : le montant). Une section d'au plus `parent_max_chars` caractères est
        jointe en entier ; au-delà (article 3 : 11 parties, fiches CNIL : jusqu'à 23), seulement
        les parties voisines. Les parties jointes suivent l'ordre du texte, et gardent le score du
        reranker s'il les a notées (None sinon).
        """
        out: list[dict] = []
        placed: set[str] = set()
        by_id = {r["id"]: r for r in rows}
        for row in rows:
            if row["id"] in placed:
                continue
            parts = self.section_parts.get((row["document_id"], row["metadata"]["label"]), [])
            ids = [p["id"] for p in parts]
            if row["id"] not in ids:  # chunk inconnu de l'index chargé : seul
                parts, ids = [row], [row["id"]]
            if sum(len(p["content"]) for p in parts) > self.settings.parent_max_chars:
                i = ids.index(row["id"])
                parts = parts[max(0, i - 1) : i + 2]
            for part in parts:
                if part["id"] in placed:
                    continue
                placed.add(part["id"])
                if part["id"] in by_id:
                    out.append(by_id[part["id"]])
                else:
                    sibling = {**part, "score": scored.get(part["id"])}
                    out.append({**sibling, "part_of": row["metadata"]["label"]})
        return out

    def with_citations(self, rows: list[dict], final_k: int, max_extra: int = 6) -> list[dict]:
        """Les final_k meilleurs chunks + les sections qu'ils citent (graphe des citations, 1 saut).

        « l'article 83, paragraphe 5, vise les articles 12 à 22 » : l'article 15 (droit d'accès)
        est joint au contexte. Quand un chunk cite plus de sections que max_extra, on garde
        celles que le reranker a le mieux notées pour la question (scores des autres lignes de
        `rows`, déjà calculés : aucun appel de plus) ; les autres suivent l'ordre du texte.
        Les sections ajoutées gardent ce score, ou None si elles n'ont pas été notées.
        """
        context = rows[:final_k]
        if self.settings.parent_max_chars:
            ids = {r["id"]: r["score"] for r in rows if r.get("score") is not None}
            context = self.with_siblings(context, ids)
        have = {(r["document_id"], r["metadata"]["label"]) for r in context}
        scored: dict[tuple[str, str], float] = {}
        for r in rows:
            key = (r["document_id"], r["metadata"]["label"])
            if r.get("score") is not None:
                scored[key] = max(scored.get(key, r["score"]), r["score"])
        cited: list[tuple[tuple[str, str], str]] = []
        for row in rows[:final_k]:  # les parties jointes ne sont pas suivies (bruit mesuré)
            for label in row["metadata"].get("cites", []):
                key = (row["document_id"], label)
                if key not in have and key in self.section_heads and key not in dict(cited):
                    cited.append((key, row["metadata"]["label"]))
        cited.sort(key=lambda c: -scored.get(c[0], float("-inf")))  # tri stable : ordre du texte
        for key, cited_by in cited[:max_extra]:
            head = self.section_heads[key]
            context.append({**head, "score": scored.get(key), "cited_by": cited_by})
        return context

    def _row(self, chunk_id: str, score: float) -> dict:
        if chunk_id not in self.chunks:  # chunk ajouté depuis le chargement : relu à la demande
            row = self.db.table("chunks").select("id, document_id, content, metadata")
            self.chunks[chunk_id] = row.eq("id", chunk_id).single().execute().data
        return {**self.chunks[chunk_id], "score": score}

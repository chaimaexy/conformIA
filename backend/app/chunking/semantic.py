"""Chunking sémantique (vidéo #10 du tutoriel), adapté aux textes juridiques.

1. on découpe chaque section en phrases ;
2. on calcule l'embedding de chaque phrase ;
3. on coupe là où deux phrases consécutives sont sémantiquement éloignées
   (distance cosinus au-dessus du percentile `breakpoint_percentile` du corpus) ;
4. garde-fous : jamais à cheval sur deux sections (la citation reste exacte),
   taille maximale `max_chars`, et fusion des fragments trop courts (< `min_chars`).

Le seuil est calculé sur TOUT le corpus (et non section par section) : une section
courte n'a pas assez de distances pour qu'un percentile local ait du sens.
"""

from collections.abc import Callable

import numpy as np

from app.chunking.common import header, paragraph_numbers, ref, sentences, split_to_fit
from app.chunking.models import Chunk
from app.ingestion.models import LegalSection

EmbedFn = Callable[[list[str]], np.ndarray]  # textes -> matrice (n, d) normalisée


def _distances(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.array([])
    return 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)


def _group(sents: list[str], dist: np.ndarray, threshold: float, max_chars: int, min_chars: int):
    groups: list[list[str]] = [[sents[0]]]
    for sentence, d in zip(sents[1:], dist, strict=True):
        size = sum(len(s) + 1 for s in groups[-1])
        if d > threshold or size + len(sentence) > max_chars:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    merged: list[list[str]] = []
    for g in groups:  # fusion des fragments trop courts avec le précédent si ça tient
        size = sum(len(s) + 1 for s in g)
        prev = sum(len(s) + 1 for s in merged[-1]) if merged else 0
        if merged and (size < min_chars or prev < min_chars) and prev + size <= max_chars:
            merged[-1].extend(g)
        else:
            merged.append(g)
    return merged


def chunk_document(
    sections: list[LegalSection],
    embed: EmbedFn,
    breakpoint_percentile: float = 90,
    max_chars: int = 2000,
    min_chars: int = 300,
) -> tuple[list[Chunk], float]:
    per_section = [
        [p for s in sentences(sec.text) for p in split_to_fit(s, max_chars)] for sec in sections
    ]
    flat = [s for sents in per_section for s in sents]
    vectors = embed(flat)

    distances, offset = [], 0
    for sents in per_section:
        distances.append(_distances(vectors[offset : offset + len(sents)]))
        offset += len(sents)
    all_distances = np.concatenate([d for d in distances if len(d)])
    threshold = float(np.percentile(all_distances, breakpoint_percentile))

    chunks: list[Chunk] = []
    for section, sents, dist in zip(sections, per_section, distances, strict=True):
        if not sents:
            continue
        groups = _group(sents, dist, threshold, max_chars, min_chars)
        for i, group in enumerate(groups):
            part = f" (partie {i + 1}/{len(groups)})" if len(groups) > 1 else ""
            body = "\n".join(group)
            chunks.append(
                Chunk(
                    source_id=section.source_id,
                    strategy="semantic",
                    chunk_index=len(chunks),
                    header=header(section) + part,
                    body=body,
                    sections=[ref(section)],
                    paragraphs=paragraph_numbers(body) if section.section_type == "article" else [],
                    source_date=section.source_date.isoformat() if section.source_date else None,
                    amended_by=section.amended_by,
                    corrected_by=section.corrected_by,
                )
            )
    return chunks, threshold

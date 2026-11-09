"""Modèles d'embedding (Mistral API ou encodeurs locaux) avec cache disque."""

from app.embeddings.base import CachedEmbedder, Embedder

# Nom de colonne pgvector associé à chaque modèle (voir la migration SQL)
COLUMNS = {"mistral-embed": "embedding", "local:BAAI/bge-m3": "embedding_bge_m3"}


def get_embedder(spec: str, cached: bool = True) -> Embedder:
    """« mistral-embed » ou « local:<id Hugging Face> »."""
    if spec == "mistral-embed":
        from app.embeddings.mistral import MistralEmbedder

        embedder = MistralEmbedder()
    elif spec.startswith("local:"):
        from app.embeddings.local import LocalEmbedder

        embedder = LocalEmbedder(spec.removeprefix("local:"))
    else:
        raise ValueError(f"Modèle d'embedding inconnu : {spec}")
    return CachedEmbedder(embedder) if cached else embedder

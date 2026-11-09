"""Écriture des chunks dans Supabase (tables `documents` et `chunks`, voir supabase/migrations).

Les écritures utilisent la clé secrète (côté serveur uniquement), qui contourne RLS.
Ré-indexer une stratégie remplace ses chunks : on supprime puis on insère par lots.
"""

import logging

import numpy as np
from supabase import Client, create_client

from app.chunking.models import Chunk
from app.core.config import get_settings
from app.embeddings import COLUMNS

logger = logging.getLogger(__name__)
BATCH = 50  # lignes par requête (2 vecteurs de 1024 floats ~ 25 Ko par ligne)


def client() -> Client:
    settings = get_settings()
    if not (settings.supabase_url and settings.supabase_secret_key):
        raise RuntimeError("SUPABASE_URL et SUPABASE_SECRET_KEY doivent être définis dans .env")
    return create_client(settings.supabase_url, settings.supabase_secret_key)


def vector_literal(vector: np.ndarray) -> str:
    """Format texte accepté par pgvector : '[0.1,0.2,...]' (6 décimales suffisent)."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def ensure_document(db: Client, source_id: str, title: str, url: str, celex: str | None) -> str:
    existing = db.table("documents").select("id").eq("source_id", source_id).execute().data
    if existing:
        db.table("documents").update({"title": title, "url": url, "celex": celex}).eq(
            "id", existing[0]["id"]
        ).execute()
        return existing[0]["id"]
    row = {"source_id": source_id, "title": title, "url": url, "celex": celex}
    return db.table("documents").insert(row).execute().data[0]["id"]


def replace_chunks(
    db: Client,
    document_id: str,
    strategy: str,
    chunks: list[Chunk],
    embeddings: dict[str, np.ndarray],  # modèle -> matrice (len(chunks), 1024)
) -> int:
    db.table("chunks").delete().eq("document_id", document_id).eq("strategy", strategy).execute()
    rows = []
    for i, chunk in enumerate(chunks):
        row = {
            "document_id": document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "metadata": chunk.metadata(),
            "strategy": strategy,
        }
        for model, matrix in embeddings.items():
            row[COLUMNS[model]] = vector_literal(matrix[i])
        rows.append(row)
    for start in range(0, len(rows), BATCH):
        db.table("chunks").insert(rows[start : start + BATCH]).execute()
    return len(rows)

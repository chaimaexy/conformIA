"""Embeddings locaux avec sentence-transformers (encodeurs de la famille BERT), sur CPU.

`BAAI/bge-m3` : XLM-RoBERTa large affiné pour la recherche, multilingue (dont le français),
licence MIT, vecteurs de 1024 dimensions. Aucun préfixe d'instruction n'est requis.
Dépendances optionnelles : pip install -e ".[local]"
"""

import logging
from pathlib import Path

import numpy as np

from app.core.config import DATA_DIR, get_settings

logger = logging.getLogger(__name__)
MODELS_CACHE = DATA_DIR / "cache" / "hf"  # dans le dépôt (ignoré par git)
HOME_CACHE = Path.home() / ".cache" / "conformia" / "hf"  # repli quand le disque du dépôt est plein


def models_cache(model_id: str) -> str:
    """Dossier du cache Hugging Face à utiliser pour ce modèle.

    Le dossier où ses poids sont déjà téléchargés, sinon MODELS_CACHE_DIR (.env), sinon
    data/cache/hf. Les modèles peuvent ainsi être répartis sur plusieurs disques.
    """
    configured = get_settings().models_cache_dir
    folder = "models--" + model_id.replace("/", "--")
    for root in (configured, MODELS_CACHE, HOME_CACHE):
        snapshots = Path(root) / folder / "snapshots" if root else None
        if snapshots and any(snapshots.glob("*/*.safetensors")):
            return str(root)
    return str(configured or MODELS_CACHE)


class LocalEmbedder:
    dimension = 1024

    def __init__(
        self, model_id: str = "BAAI/bge-m3", max_seq_length: int = 1024, batch_size: int = 8
    ):
        self.model_id = model_id
        self.name = f"local:{model_id}"
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # import lourd : à la demande

            logger.info("Chargement de %s (CPU)...", self.model_id)
            self._model = SentenceTransformer(
                self.model_id, device="cpu", cache_folder=models_cache(self.model_id)
            )
            self._model.max_seq_length = self.max_seq_length
            m = self._model  # la méthode a été renommée dans sentence-transformers 5
            if hasattr(m, "get_embedding_dimension"):
                self.dimension = m.get_embedding_dimension()
            else:
                self.dimension = m.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        ).astype(np.float32)

"""Lazy-loaded sentence-transformer embedding model.

Usage::

    from .embeddings import encode, embeddings_available

    if embeddings_available():
        vec = encode(["how do we authenticate users?"])
"""

from __future__ import annotations

from typing import Any

_MODEL: Any = None
_MODEL_NAME = "all-MiniLM-L6-v2"
_AVAILABLE: bool | None = None


def embeddings_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def _load() -> Any:
    global _MODEL
    if _MODEL is None:
        if not embeddings_available():
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install vouch[embeddings]"
            )
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def encode(texts: list[str], *, normalize: bool = True):
    import numpy as np
    model = _load()
    emb = model.encode(texts, normalize_embeddings=normalize)
    return np.asarray(emb, dtype=np.float32)

"""Tests for embedding-based semantic search."""

from __future__ import annotations

from pathlib import Path

import pytest

from vouch import index_db
from vouch.embeddings import embeddings_available, encode
from vouch.models import Claim, Page
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def test_embeddings_available_returns_bool() -> None:
    result = embeddings_available()
    assert isinstance(result, bool)


def test_encode_returns_expected_shape() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    vec = encode(["hello world"])
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 384)
    assert vec.dtype == np.float32


def test_encode_multiple_texts() -> None:
    pytest.importorskip("sentence_transformers")
    vecs = encode(["hello", "world", "foo bar"])
    assert vecs.shape == (3, 384)


def test_index_and_search_embedding(store: KBStore) -> None:
    pytest.importorskip("sentence_transformers")
    store.put_claim(Claim(id="c1", text="login flow uses session cookies"))
    store.put_claim(Claim(id="c2", text="the sky is blue today"))
    store.put_page(Page(id="p1", title="Auth docs", body="how we authenticate users"))

    with index_db.open_db(store.kb_dir) as conn:
        for c in store.list_claims():
            vec = encode([c.text])[0].tolist()
            index_db.index_embedding(conn, kind="claim", id=c.id, vec=vec)
        for p in store.list_pages():
            page_vec = encode([f"{p.title} {p.body}"])[0].tolist()
            index_db.index_embedding(conn, kind="page", id=p.id, vec=page_vec)

    query_vec = encode(["how do we authenticate users?"])[0].tolist()
    hits = index_db.search_embeddings(store.kb_dir, query_vec, limit=5)

    assert len(hits) >= 1
    kinds = {h[0] for h in hits}
    assert "claim" in kinds or "page" in kinds


def test_search_relevant_before_irrelevant(store: KBStore) -> None:
    pytest.importorskip("sentence_transformers")
    store.put_claim(Claim(id="c1", text="login flow uses session cookies signed by API"))
    store.put_claim(Claim(id="c2", text="the weather is nice today"))

    with index_db.open_db(store.kb_dir) as conn:
        for c in store.list_claims():
            vec = encode([c.text])[0].tolist()
            index_db.index_embedding(conn, kind="claim", id=c.id, vec=vec)

    query_vec = encode(["how do we authenticate users?"])[0].tolist()
    hits = index_db.search_embeddings(store.kb_dir, query_vec, limit=5)

    assert len(hits) >= 2
    assert hits[0][1] == "c1", "semantically relevant claim should rank first"


def test_empty_index_returns_empty(store: KBStore) -> None:
    pytest.importorskip("numpy")
    query_vec = [0.0] * 384
    hits = index_db.search_embeddings(store.kb_dir, query_vec, limit=5)
    assert hits == []

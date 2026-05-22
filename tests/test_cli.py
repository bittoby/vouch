"""CLI surface — every command must turn domain errors into a clean
`Error: ...` line via click.ClickException, never a raw Python traceback.

These regressions cover the bug class where CLI handlers caught
``(ArtifactNotFoundError, ValueError)`` but ``proposals.approve()`` /
``proposals.reject()`` (and the ``propose_*`` helpers) raise
``ProposalError`` (a ``RuntimeError`` subclass) for validation failures —
which slipped past the except and surfaced as a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch.cli import cli
from vouch.models import Claim
from vouch.proposals import propose_claim
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path)
    monkeypatch.chdir(s.root)
    return s


def _assert_clean_error(result, needle: str) -> None:
    assert result.exit_code != 0, result.output
    # click.ClickException renders as ``Error: <msg>``; a raw traceback would
    # include the Python frame marker ``Traceback (most recent call last)``.
    assert "Traceback" not in result.output, result.output
    assert "Error:" in result.output, result.output
    assert needle in result.output, result.output


def test_approve_already_decided_proposal_shows_clean_error(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    runner = CliRunner()
    first = runner.invoke(cli, ["approve", pr.id])
    assert first.exit_code == 0, first.output
    second = runner.invoke(cli, ["approve", pr.id])
    _assert_clean_error(second, "not pending")


def test_reject_empty_reason_shows_clean_error(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    result = CliRunner().invoke(cli, ["reject", pr.id, "--reason", "   "])
    _assert_clean_error(result, "reason")


def test_propose_claim_empty_text_shows_clean_error(store: KBStore) -> None:
    src = store.put_source(b"e")
    result = CliRunner().invoke(
        cli, ["propose-claim", "--text", "   ", "--source", src.id]
    )
    _assert_clean_error(result, "claim text")


def test_propose_claim_unknown_source_shows_clean_error(store: KBStore) -> None:
    result = CliRunner().invoke(
        cli, ["propose-claim", "--text", "ok", "--source", "deadbeef"]
    )
    _assert_clean_error(result, "unknown source")


def test_propose_entity_empty_name_shows_clean_error(store: KBStore) -> None:
    result = CliRunner().invoke(
        cli, ["propose-entity", "--name", "   ", "--type", "project"]
    )
    _assert_clean_error(result, "entity name")


def test_show_missing_proposal_shows_clean_error(store: KBStore) -> None:
    result = CliRunner().invoke(cli, ["show", "no-such-proposal"])
    _assert_clean_error(result, "proposal no-such-proposal")


def test_search_fts5_backend_label(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vouch search prints (fts5) when FTS5 returns hits."""
    from vouch import index_db
    from vouch.proposals import approve as do_approve
    from vouch.proposals import propose_claim
    src = store.put_source(b"e")
    pr = propose_claim(store, text="findable token", evidence=[src.id], proposed_by="agent")
    do_approve(store, pr.id, approved_by="reviewer")
    # Index only the FTS5 tables directly — no embedding stack needed
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_claim(
            conn, id="c-findable", text="findable token",
            type="observation", status="actionable", tags=[],
        )
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "findable"])
    assert result.exit_code == 0, result.output
    assert "(fts5)" in result.output


def test_search_substring_backend_label(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vouch search prints (substring) when FTS5 raises and fallback runs."""
    from vouch.proposals import approve as do_approve
    from vouch.proposals import propose_claim
    src = store.put_source(b"e")
    pr = propose_claim(store, text="findable token", evidence=[src.id], proposed_by="agent")
    do_approve(store, pr.id, approved_by="reviewer")
    # Remove state.db so FTS5 raises and substring fallback runs
    state_db = store.kb_dir / "state.db"
    if state_db.exists():
        state_db.unlink()
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "findable"])
    assert result.exit_code == 0, result.output
    assert "(substring)" in result.output


def test_status_uses_color_in_human_output(store: KBStore) -> None:
    result = CliRunner().invoke(cli, ["status"], color=True)

    assert result.exit_code == 0, result.output
    assert "KB status" in result.output
    assert "\x1b[" in result.output


def test_lint_json_is_machine_readable(store: KBStore) -> None:
    result = CliRunner().invoke(cli, ["lint", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["findings"] == []


def test_search_json_includes_backend(store: KBStore) -> None:
    from vouch import index_db

    src = store.put_source(b"e")
    store.put_claim(
        Claim(id="c-findable", text="findable token", evidence=[src.id])
    )
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_claim(
            conn, id="c-findable", text="findable token",
            type="observation", status="working", tags=[],
        )

    result = CliRunner().invoke(cli, ["search", "findable", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["id"] == "c-findable"
    assert payload[0]["backend"] == "fts5"


def test_export_defaults_to_human_output_and_supports_json(
    store: KBStore, tmp_path: Path
) -> None:
    out = tmp_path / "kb.tar.gz"

    human = CliRunner().invoke(cli, ["export", "--out", str(out)])

    assert human.exit_code == 0, human.output
    assert "bundle exported" in human.output
    assert out.exists()

    machine = CliRunner().invoke(
        cli, ["export", "--out", str(tmp_path / "kb2.tar.gz"), "--json"]
    )

    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["files"] >= 1

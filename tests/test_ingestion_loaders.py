"""Tests for the Obsidian/Markdown corpus loader using a tiny fake vault.

A small vault is written under pytest ``tmp_path`` and ``load_corpus`` is exercised for
title/tags/wikilinks/frontmatter/rel_path correctness, determinism, the ``.obsidian`` skip,
malformed frontmatter resilience, and graceful skipping of optional binary formats. No test
imports ``yaml``/``pymupdf``/``docx``; the YAML fallback is exercised by forcing an
ImportError via monkeypatch so the suite passes regardless of which optional deps exist.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from rag.ingestion.loaders import load_corpus

NOTE_MD = """\
---
title: My First Note
tags: [alpha, beta]
---
# Heading One

Body text with an inline #gamma tag and a [[Other Note|alias]] plus [[Plain]].
"""

NESTED_MD = """\
# Nested Title

No frontmatter here; the title comes from the H1.
"""

WEIRD_MD = """\
---
title: Broken
no closing fence here, this is just body content.

Still body, the document must not be consumed.
"""

PLAIN_TXT = "Just a plain text file.\nSecond line."


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Write a tiny Obsidian-style vault and return its root."""
    (tmp_path / "note.md").write_text(NOTE_MD, encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text(NESTED_MD, encoding="utf-8")
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    (obsidian / "app.json").write_text('{"theme": "obsidian"}', encoding="utf-8")
    (tmp_path / "plain.txt").write_text(PLAIN_TXT, encoding="utf-8")
    (tmp_path / "weird.md").write_text(WEIRD_MD, encoding="utf-8")
    return tmp_path


def _by_rel(docs: list, rel_path: str):
    return next(d for d in docs if d.rel_path == rel_path)


def test_loads_all_text_files(vault: Path) -> None:
    docs = load_corpus(vault)
    rel_paths = {d.rel_path for d in docs}
    assert rel_paths == {"note.md", "sub/nested.md", "plain.txt", "weird.md"}
    assert all(".obsidian" not in d.rel_path for d in docs)


def test_title_precedence(vault: Path) -> None:
    docs = load_corpus(vault)
    assert _by_rel(docs, "note.md").title == "My First Note"  # frontmatter wins
    assert _by_rel(docs, "sub/nested.md").title == "Nested Title"  # H1 fallback
    assert _by_rel(docs, "plain.txt").title == "plain"  # stem fallback


def test_tags_merged_sorted(vault: Path) -> None:
    note = _by_rel(load_corpus(vault), "note.md")
    assert note.metadata["tags"] == ["alpha", "beta", "gamma"]  # frontmatter + inline, sorted


def test_wikilinks_extracted(vault: Path) -> None:
    note = _by_rel(load_corpus(vault), "note.md")
    assert note.metadata["wikilinks"] == ["Other Note", "Plain"]  # targets only, sorted


def test_wikilink_render_in_body(vault: Path) -> None:
    note = _by_rel(load_corpus(vault), "note.md")
    assert "alias" in note.text  # alias is the visible text for [[Other Note|alias]]
    assert "Plain" in note.text
    assert "[[" not in note.text  # brackets removed
    assert "]]" not in note.text


def test_frontmatter_recorded(vault: Path) -> None:
    note = _by_rel(load_corpus(vault), "note.md")
    assert note.metadata["frontmatter"]["title"] == "My First Note"
    assert note.metadata["source_format"] == "markdown"


def test_rel_path_posix(vault: Path) -> None:
    docs = load_corpus(vault)
    # Forward slash even on Windows -> ids match across OS.
    assert any(d.rel_path == "sub/nested.md" for d in docs)


def test_folder_metadata(vault: Path) -> None:
    docs = load_corpus(vault)
    assert _by_rel(docs, "sub/nested.md").metadata["folder"] == "sub"
    assert _by_rel(docs, "note.md").metadata["folder"] == ""


def test_deterministic_order(vault: Path) -> None:
    first = [d.rel_path for d in load_corpus(vault)]
    second = [d.rel_path for d in load_corpus(vault)]
    assert first == second
    assert first == sorted(first)


def test_malformed_frontmatter_no_crash(vault: Path) -> None:
    weird = _by_rel(load_corpus(vault), "weird.md")
    assert weird.metadata["frontmatter"] == {}  # no closing fence -> not frontmatter
    assert "Still body" in weird.text  # body preserved, document not consumed
    assert weird.text.startswith("---")  # the leading fence is part of the body


def test_doc_id_from_rel_path(vault: Path) -> None:
    docs = load_corpus(vault)
    for doc in docs:
        expected = hashlib.sha256(doc.rel_path.encode("utf-8")).hexdigest()[:16]
        assert doc.doc_id == expected


def test_yaml_fallback_used_when_yaml_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force an ImportError on ``import yaml`` to exercise the minimal fallback parser."""
    (tmp_path / "fm.md").write_text(
        "---\ntitle: Fallback Title\ntags: [x, y]\n---\nBody.\n", encoding="utf-8"
    )
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "yaml":
            raise ImportError("yaml disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    doc = load_corpus(tmp_path)[0]
    assert doc.title == "Fallback Title"
    assert doc.metadata["frontmatter"]["title"] == "Fallback Title"
    assert doc.metadata["tags"] == ["x", "y"]  # parsed by the fallback, then sorted


def test_frontmatter_dates_are_json_safe(tmp_path: Path) -> None:
    """Unquoted Obsidian dates/datetimes must not leave non-JSON types in metadata.

    PyYAML decodes ``created: 2024-01-15`` to ``datetime.date`` and
    ``modified: 2024-01-15 10:30:00`` to ``datetime.datetime``; without coercion
    ``json.dumps(doc.metadata)`` raises ``TypeError``. Requires PyYAML to reproduce the
    date-decoding behavior, so it is skipped if PyYAML is absent.
    """
    pytest.importorskip("yaml")
    (tmp_path / "dated.md").write_text(
        "---\n"
        "title: Dated Note\n"
        "created: 2024-01-15\n"
        "modified: 2024-01-15 10:30:00\n"
        "tags: [a, b]\n"
        "---\n"
        "# Dated\nBody.\n",
        encoding="utf-8",
    )
    doc = load_corpus(tmp_path)[0]
    # The contract: the whole metadata dict round-trips through json.dumps without raising.
    serialized = json.dumps(doc.metadata)
    assert "2024-01-15" in serialized
    fm = doc.metadata["frontmatter"]
    assert isinstance(fm["created"], str)
    assert fm["created"] == "2024-01-15"
    assert isinstance(fm["modified"], str)
    assert fm["modified"].startswith("2024-01-15T10:30:00")


def test_frontmatter_tags_as_mapping(tmp_path: Path) -> None:
    """A ``tags:`` nested mapping flattens to its keys, not a single ``str(dict)`` pseudo-tag."""
    pytest.importorskip("yaml")
    (tmp_path / "mapped.md").write_text(
        "---\ntitle: Mapped\ntags:\n  project: x\n  status: y\n---\nBody.\n",
        encoding="utf-8",
    )
    doc = load_corpus(tmp_path)[0]
    assert doc.metadata["tags"] == ["project", "status"]  # keys flattened, sorted
    # No garbage "{'project': 'x', ...}" tag leaked in.
    assert all("{" not in tag for tag in doc.metadata["tags"])


def test_title_skips_h1_inside_code_fence(tmp_path: Path) -> None:
    """A ``# comment`` inside a fenced code block must not become the document title."""
    (tmp_path / "code.md").write_text(
        "```\n# This is code not a title\n```\n# Real Title\nBody.\n",
        encoding="utf-8",
    )
    doc = load_corpus(tmp_path)[0]
    assert doc.title == "Real Title"


def test_pdf_skipped_gracefully_when_dep_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .pdf must be skipped (not crash) when PyMuPDF is unavailable.

    The missing-dep path is forced via monkeypatch so the test holds regardless of whether
    ``pymupdf`` happens to be installed in the test environment.
    """
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "fitz":
            raise ImportError("pymupdf disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    (tmp_path / "ok.md").write_text("# Title\nbody", encoding="utf-8")
    docs = load_corpus(tmp_path)  # must not raise
    rel_paths = {d.rel_path for d in docs}
    assert "doc.pdf" not in rel_paths
    assert "ok.md" in rel_paths


def test_docx_skipped_gracefully_when_dep_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .docx must be skipped (not crash) when python-docx is unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "docx":
            raise ImportError("python-docx disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    (tmp_path / "report.docx").write_bytes(b"PK\x03\x04 not a real docx")
    (tmp_path / "ok.md").write_text("# Title\nbody", encoding="utf-8")
    docs = load_corpus(tmp_path)  # must not raise
    rel_paths = {d.rel_path for d in docs}
    assert "report.docx" not in rel_paths
    assert "ok.md" in rel_paths

"""Evidence-table integrity: every row traces to a paper, and 'verified' means the source was actually read."""

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIORS = yaml.safe_load((ROOT / "data" / "priors.yaml").read_text())
BIB_KEYS = set(re.findall(r"^@\w+\{([^,]+),", (ROOT / "refs" / "references.bib").read_text(), flags=re.M))
READ_LEVELS = {"table", "text"}


def _rows(node):
    """Yield every dict that carries an id, at any depth."""
    if isinstance(node, dict):
        if "id" in node:
            yield node
        for v in node.values():
            yield from _rows(v)
    elif isinstance(node, list):
        for v in node:
            yield from _rows(v)


ROWS = list(_rows(PRIORS))


def test_there_are_rows():
    assert len(ROWS) > 40


def test_bibliography_has_the_folder_articles_and_the_retrieved_open_access_papers():
    # 57 articles in DATA_Articles, 15 free papers retrieved on request, 1 supplied by the user
    assert len(BIB_KEYS) == 57 + 15 + 1


@pytest.mark.parametrize("row", [r for r in ROWS if "bib" in r], ids=lambda r: r["id"])
def test_cited_key_exists_in_bibliography(row):
    assert row["bib"] in BIB_KEYS, f"{row['id']} cites {row['bib']}, which is not in refs/references.bib"


@pytest.mark.parametrize("row", [r for r in ROWS if "verified" in r], ids=lambda r: r["id"])
def test_verified_only_when_source_was_read(row):
    if row["verified"]:
        assert row.get("checked") in READ_LEVELS, f"{row['id']} is marked verified without reading a table or the text"
        assert "bib" in row, f"{row['id']} is marked verified but cites no bibliography key"


@pytest.mark.parametrize("row", [r for r in ROWS if r.get("checked") in READ_LEVELS], ids=lambda r: r["id"])
def test_rows_read_from_a_paper_are_marked_verified(row):
    assert row.get("verified") is True

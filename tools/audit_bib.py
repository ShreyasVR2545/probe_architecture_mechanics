"""Audit references.bib against authoritative arXiv metadata.

Motivation: a bibliography is a set of factual claims, and search-engine snippets are
not evidence. This queries the arXiv API directly for every arXiv ID in the bib and
reports, per entry:

  * the canonical title (so Title Case and wording can be checked),
  * the FULL author list (the bib must not truncate with "et al."),
  * any journal-ref / DOI, which is the authoritative signal that a preprint has been
    accepted somewhere and the entry should be upgraded to @inproceedings.

Entries with no arXiv ID (books, classic papers) are listed as unchecked rather than
silently passed.

  python tools/audit_bib.py
"""
from __future__ import annotations

import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "references.bib"
API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Deliberate divergences from the arXiv metadata, with the reason. Anything NOT listed
# here is reported: the point of the audit is that an unexplained mismatch is loud.
KNOWN_TITLE_DIFFS = {
    "goldowskydill2025deception":
        "we cite the ICML proceedings title ('with'), arXiv preprint reads 'Using'",
}

RE_ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)
RE_FIELD = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", re.S)
# Matches both "arXiv:2502.03407" inside a journal field and a bare eprint = {2502.03407}.
# Published entries carry the id only in `eprint`; without the second alternative they
# would silently drop out of the audit at exactly the moment they were upgraded.
RE_ARXIV = re.compile(
    r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5})|eprint\s*=\s*\{\s*([0-9]{4}\.[0-9]{4,5})", re.I)


def parse_bib(text: str) -> list[dict]:
    out = []
    for kind, key, body in RE_ENTRY.findall(text):
        fields = {k.lower(): " ".join(v.split()) for k, v in RE_FIELD.findall(body + "\n")}
        blob = kind + key + body
        m = RE_ARXIV.search(blob)
        aid = (m.group(1) or m.group(2)) if m else None
        out.append({"kind": kind, "key": key, "fields": fields, "arxiv": aid})
    return out


def fetch(ids: list[str]) -> dict[str, dict]:
    """One batched arXiv API call. Returns {id: metadata}."""
    if not ids:
        return {}
    q = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        root = ET.fromstring(r.read())
    meta: dict[str, dict] = {}
    for e in root.findall("a:entry", NS):
        raw = e.findtext("a:id", "", NS)
        aid = raw.rsplit("/", 1)[-1].split("v")[0]
        meta[aid] = {
            "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
            "authors": [a.findtext("a:name", "", NS)
                        for a in e.findall("a:author", NS)],
            "journal_ref": (e.findtext("arxiv:journal_ref", "", NS) or "").strip(),
            "doi": (e.findtext("arxiv:doi", "", NS) or "").strip(),
            "comment": " ".join((e.findtext("arxiv:comment", "", NS) or "").split()),
        }
    return meta


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    ids = [e["arxiv"] for e in entries if e["arxiv"]]
    print(f"bib entries : {len(entries)}")
    print(f"with arXiv  : {len(ids)}")
    print(f"querying arXiv API for {len(ids)} ids ...\n")
    try:
        meta = fetch(ids)
    except Exception as exc:
        print(f"  API unreachable ({type(exc).__name__}); cannot audit offline.")
        return 2
    time.sleep(0.2)

    upgrades, mismatches, unchecked, ok, expected = [], [], [], [], []
    for e in entries:
        key, f, aid = e["key"], e["fields"], e["arxiv"]
        if not aid:
            unchecked.append(key)
            continue
        m = meta.get(aid)
        if not m:
            unchecked.append(f"{key} (id {aid} not returned)")
            continue

        # venue upgrade available?
        has_venue = e["kind"].lower() in ("inproceedings", "incollection", "article") and \
            any(k in f for k in ("booktitle", "journal")) and \
            "arXiv preprint" not in f.get("journal", "")
        if m["journal_ref"] and not has_venue:
            upgrades.append((key, aid, m["journal_ref"]))

        # title agreement, ignoring case/punctuation/braces
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())
        if norm(f.get("title", "")) != norm(m["title"]):
            if key in KNOWN_TITLE_DIFFS:
                expected.append((key, KNOWN_TITLE_DIFFS[key]))
            else:
                mismatches.append((key, f.get("title", ""), m["title"]))

        if "others" in f.get("author", "") or "et al" in f.get("author", "").lower():
            mismatches.append((key, "TRUNCATED AUTHOR LIST",
                               "; ".join(m["authors"])))
        if not upgrades or upgrades[-1][0] != key:
            ok.append(key)

    print("=" * 78)
    print(f"VENUE UPGRADES AVAILABLE ({len(upgrades)})")
    print("=" * 78)
    for key, aid, jr in upgrades:
        print(f"  {key:28s} arXiv:{aid}\n      -> {jr}")
    if not upgrades:
        print("  none")

    print(f"\n{'=' * 78}\nTITLE / AUTHOR DISCREPANCIES ({len(mismatches)})\n{'=' * 78}")
    for key, mine, theirs in mismatches:
        print(f"  {key}\n      bib   : {mine}\n      arXiv : {theirs}")
    if not mismatches:
        print("  none")

    print(f"\n{'=' * 78}\nDOCUMENTED EXCEPTIONS ({len(expected)})\n{'=' * 78}")
    for key, why in expected:
        print(f"  {key}\n      {why}")
    if not expected:
        print("  none")

    print(f"\n{'=' * 78}\nNOT ON arXiv, unchecked here ({len(unchecked)})\n{'=' * 78}")
    print("  " + ", ".join(unchecked) if unchecked else "  none")

    print(f"\nRESULT: {len(upgrades)} upgrade(s), {len(mismatches)} discrepancy(ies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

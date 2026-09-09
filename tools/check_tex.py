"""Structural validation and numbering audit for the LaTeX sources.

Audits BOTH manuscripts (paper.tex and math_formulation.tex). Everything here is
checkable without invoking a compiler: environment balance, brace balance, math-mode
parity, label uniqueness, dangling \\ref targets, the amsthm number each labelled result
will receive, and -- for files with a bibliography -- dangling \\cite keys, duplicate
BibTeX keys, and entries that are defined but never cited.

It is not a substitute for a compile, and says so in its own output. Run:

    python tools/check_tex.py              # both files
    python tools/check_tex.py paper.tex    # one file
"""
from __future__ import annotations

import collections
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ["paper.tex", "math_formulation.tex"]
BIB = ROOT / "references.bib"

RE_BEGIN = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
RE_END = re.compile(r"\\end\{([a-zA-Z*]+)\}")
RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(?:eq|c|C|auto)?ref\{([^}]+)\}")
RE_CITE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}")
RE_STRUCT = re.compile(r"\\(section|subsection)\{|\\begin\{([a-zA-Z*]+)\}")
RE_BIBKEY = re.compile(r"^@\w+\{([^,\s]+)\s*,", re.M)

# amsthm environments sharing one counter, reset per section
THM_ENVS = {"theorem", "proposition", "corollary", "remark", "assumption",
            "lemma", "definition"}
COMPILERS = ("latexmk", "pdflatex", "xelatex", "lualatex", "tectonic")


def strip_comments(src: str) -> str:
    """Drop comment lines and trailing comments, respecting escaped percent signs."""
    out = []
    for line in src.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(out)


def numbering(src: str) -> list[tuple[str, str, str]]:
    """Return (number, env, label) for each theorem-like environment, in order."""
    out: list[tuple[str, str, str]] = []
    sec = ctr = 0
    for m in RE_STRUCT.finditer(src):
        if m.group(1) == "section":
            sec += 1
            ctr = 0
            continue
        if m.group(1) == "subsection":
            continue
        env = m.group(2)
        if env not in THM_ENVS:
            continue
        ctr += 1
        tail = src[m.end():m.end() + 300]
        lab = RE_LABEL.search(tail)
        out.append((f"{sec}.{ctr}", env, lab.group(1) if lab else "(unlabelled)"))
    return out


def bib_keys() -> tuple[list[str], list[str]]:
    """(all keys in order, duplicated keys)."""
    if not BIB.exists():
        return [], []
    keys = RE_BIBKEY.findall(BIB.read_text(encoding="utf-8"))
    dupes = [k for k, c in collections.Counter(keys).items() if c > 1]
    return keys, dupes


def check(path: Path, keys: list[str], dupes: list[str]) -> bool:
    src = path.read_text(encoding="utf-8")
    body = strip_comments(src)

    cb = collections.Counter(RE_BEGIN.findall(body))
    ce = collections.Counter(RE_END.findall(body))
    unbalanced = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}

    braces_ok = body.count("{") == body.count("}")
    dollars = body.count("$")
    labels = RE_LABEL.findall(body)
    refs = RE_REF.findall(body)
    lab_dupes = [l for l, c in collections.Counter(labels).items() if c > 1]
    dangling_refs = sorted(set(refs) - set(labels))

    # citations, only meaningful for a file that carries a \bibliography
    has_bib = "\\bibliography{" in body
    cited = sorted({k.strip() for grp in RE_CITE.findall(body)
                    for k in grp.split(",") if k.strip()})
    dangling_cites = sorted(set(cited) - set(keys)) if has_bib else []
    uncited = sorted(set(keys) - set(cited)) if has_bib else []

    print("=" * 78)
    print(f"file        : {path.name}  ({len(src.splitlines())} lines)")
    print("=" * 78)
    print(f"environments: {len(cb)} kinds, {sum(cb.values())} opens")
    print(f"unbalanced  : {unbalanced or 'none'}")
    print(f"braces      : {body.count('{')} open / {body.count('}')} close  "
          f"{'OK' if braces_ok else 'MISMATCH'}")
    print(f"inline math : {dollars} '$' {'(even)' if dollars % 2 == 0 else '(ODD - unclosed)'}")
    print(f"labels      : {len(labels)} unique={len(set(labels))}  "
          f"duplicates: {lab_dupes or 'none'}")
    print(f"\\ref targets: {len(refs)} used, dangling: {dangling_refs or 'none'}")

    if has_bib:
        print(f"bib entries : {len(keys)} (duplicate keys: {dupes or 'none'})")
        print(f"\\cite keys  : {len(cited)} used, dangling: {dangling_cites or 'none'}")
        print(f"uncited     : {uncited or 'none'}")

    print("\namsthm numbering (shared counter, reset per section):")
    for num, env, lab in numbering(body):
        print(f"  {num:>6s}  {env:<12s} {lab}")

    ok = (not unbalanced and braces_ok and dollars % 2 == 0
          and not lab_dupes and not dangling_refs
          and not dangling_cites and not uncited and not dupes)
    print(f"\nRESULT      : {'STRUCTURALLY VALID' if ok else 'PROBLEMS FOUND'}\n")
    return ok


def main(argv: list[str]) -> int:
    names = argv[1:] or DEFAULT_TEX
    keys, dupes = bib_keys()
    results = {}
    for name in names:
        p = ROOT / name
        if not p.exists():
            print(f"missing: {p}")
            results[name] = False
            continue
        results[name] = check(p, keys, dupes)

    found = [c for c in COMPILERS if shutil.which(c)]
    print(f"LaTeX toolchain: {found if found else 'NONE FOUND on this host'}")
    print(f"  searched: {', '.join(COMPILERS)}")
    if not found:
        print("  note: structural check only -- no compiler here, so this is NOT a compile.")

    bad = [n for n, v in results.items() if not v]
    print(f"\nOVERALL     : {'ALL VALID' if not bad else 'FAILED: ' + ', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

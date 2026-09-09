"""Structural validation and numbering audit for math_formulation.tex.

No LaTeX toolchain is installed on this host (checked: pdflatex, xelatex, lualatex,
latexmk, tectonic all absent), so this checks everything that can be checked without
one: environment balance, brace balance, math-mode parity, label uniqueness, dangling
references, and the amsthm numbering each labelled result will receive.

It is not a substitute for a compile, and says so in its own output.
"""
from __future__ import annotations

import collections
import re
import shutil
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "math_formulation.tex"

RE_BEGIN = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
RE_END = re.compile(r"\\end\{([a-zA-Z*]+)\}")
RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")
RE_STRUCT = re.compile(r"\\(section|subsection)\{|\\begin\{([a-zA-Z*]+)\}")

# amsthm environments sharing one counter, reset per section
THM_ENVS = {"theorem", "proposition", "corollary", "remark", "assumption",
            "lemma", "definition"}
COMPILERS = ("pdflatex", "xelatex", "lualatex", "latexmk", "tectonic")


def numbering(src: str) -> list[tuple[str, str, str]]:
    """Return (number, env, label) for each theorem-like environment, in order."""
    out: list[tuple[str, str, str]] = []
    sec = 0
    ctr = 0
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


def main() -> int:
    src = TEX.read_text(encoding="utf-8")
    body = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))

    cb = collections.Counter(RE_BEGIN.findall(body))
    ce = collections.Counter(RE_END.findall(body))
    unbalanced = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}

    braces_ok = body.count("{") == body.count("}")
    dollars = body.count("$")
    labels = RE_LABEL.findall(body)
    refs = RE_REF.findall(body)
    dupes = [l for l, c in collections.Counter(labels).items() if c > 1]
    dangling = sorted(set(refs) - set(labels))

    print(f"file        : {TEX.name}  ({len(src.splitlines())} lines)")
    print(f"environments: {dict(cb)}")
    print(f"unbalanced  : {unbalanced or 'none'}")
    print(f"braces      : {body.count('{')} open / {body.count('}')} close  "
          f"{'OK' if braces_ok else 'MISMATCH'}")
    print(f"inline math : {dollars} '$' {'(even)' if dollars % 2 == 0 else '(ODD - unclosed)'}")
    print(f"labels      : {len(labels)} unique={len(set(labels))}  duplicates: {dupes or 'none'}")
    print(f"references  : {len(refs)}  dangling: {dangling or 'none'}")

    print("\namsthm numbering (shared counter, reset per section):")
    nums = numbering(body)
    for num, env, lab in nums:
        print(f"  {num:>6s}  {env:<12s} {lab}")

    found = [c for c in COMPILERS if shutil.which(c)]
    print(f"\nLaTeX toolchain: {found if found else 'NONE FOUND on this host'}")
    print(f"  searched: {', '.join(COMPILERS)}")
    print("  preamble required: amsmath, amssymb, amsthm, mathtools")

    ok = (not unbalanced) and braces_ok and dollars % 2 == 0 and not dupes and not dangling
    print(f"\nRESULT      : {'STRUCTURALLY VALID' if ok else 'PROBLEMS FOUND'}")
    if not found:
        print("note        : structural check only -- no compiler on this host, so this "
              "is NOT a compile.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

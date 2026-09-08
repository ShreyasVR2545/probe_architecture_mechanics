"""Structural validation for math_formulation.tex.

No LaTeX toolchain is installed on this machine, so this checks what can be checked
without one: environment balance, brace balance, math-mode parity, label uniqueness and
dangling references. It is not a substitute for a compile.
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "math_formulation.tex"

RE_BEGIN = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
RE_END = re.compile(r"\\end\{([a-zA-Z*]+)\}")
RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")


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

    used = sorted({m for m in ("boxed", "argmin", "operatorname", "mathbb", "mathcal",
                               "varsigma", "xrightarrow", "boldsymbol", "coloneqq")
                   if re.search(r"\\" + m + r"\b", body)})
    print(f"macros      : {used}")
    print(f"preamble    : requires amsmath, amssymb, amsthm, mathtools")

    ok = (not unbalanced) and braces_ok and dollars % 2 == 0 and not dupes and not dangling
    print(f"\nRESULT      : {'STRUCTURALLY VALID' if ok else 'PROBLEMS FOUND'}")
    print("note        : structural check only -- no LaTeX toolchain on this host, "
          "so this is not a compile.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

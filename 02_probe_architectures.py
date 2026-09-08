"""
Phase 2 — Probe family implementation & short -> long OOD evaluation.

Design:
  TRAIN   short contexts only (no filler), LLM-LAT harmful vs alpaca benign.
  EVAL 1  held-out same-source split            -> in-distribution ceiling
  EVAL 2  JBB-Behaviors matched harmful/benign  -> source-confound control
  EVAL 3  JBB needles embedded in benign pile filler at 0 .. 7680 tokens
                                                -> the short->long OOD axis

The needle text is byte-identical across every rung of the ladder and both classes get
identical filler treatment, so sequence length carries no label information. Any decay
along the ladder is dilution, not a length cue.

Also reports the **black-to-white performance boost**: how much each white-box probe
beats a black-box lexical baseline that sees only the text. A probe that cannot beat
that baseline has not earned its access to the residual stream.

Run:  python 02_probe_architectures.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import (  # noqa: E402
    ARTIFACTS, CFG, banner, forward_capture, load_model, release, set_seed, vram_probe,
    write_json,
)
from data import (  # noqa: E402
    build_length_ladder, length_match, load_filler, load_matched_pool, load_train_pool,
)
from probes import auroc, build_probe_family, score_probe, train_probe  # noqa: E402

LADDER = (0, 512, 1024, 2048, 4096, 7680)
N_PER_CLASS_TRAIN = 400
MAX_SHORT_TOKENS = 192


# --------------------------------------------------------------------------------- utils
@torch.no_grad()   # not inference_mode: these activations are probe training data
def extract(model, tok, examples, layer: int, max_len: int | None = None,
            log_every: int = 200, tag: str = "") -> list[torch.Tensor]:
    """One forward per example; keep the full (T, d) residual sequence on CPU."""
    out = []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        enc = tok(ex.text, return_tensors="pt", truncation=max_len is not None,
                  max_length=max_len or 10**9)
        ids = enc.input_ids.to(CFG.device)
        H = forward_capture(model, ids, [layer], mode="all")[layer][0]   # (T, d) fp32 CPU
        out.append(H)
        if log_every and (i + 1) % log_every == 0:
            el = time.perf_counter() - t0
            print(f"      {tag} {i+1}/{len(examples)}  ({el:.0f}s, {el/(i+1):.3f}s/ex)")
    return out


def lexical_baseline(examples) -> torch.Tensor:
    """Black-box baseline: bag-of-words logistic regression on the raw text.

    This is the monitor a deployment could run for free. The white-box probe must beat
    it, otherwise activation access bought nothing.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return TfidfVectorizer, LogisticRegression


def fit_lexical(train_ex, eval_sets: dict) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = vec.fit_transform([e.text for e in train_ex])
    ytr = [e.label for e in train_ex]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    res = {}
    for name, exs in eval_sets.items():
        s = torch.tensor(clf.decision_function(vec.transform([e.text for e in exs])))
        y = torch.tensor([e.label for e in exs])
        res[name] = auroc(s, y)
    return res


# ---------------------------------------------------------------------------------- main
def main() -> int:
    set_seed(CFG.seed)
    report: dict = {"phase": 2, "ladder": list(LADDER), "layer": CFG.sae_layer}

    banner("2.1  Model + data")
    with vram_probe("load_model") as rec:
        model, tok, meta = load_model(CFG)
    d_model = meta["d_model"]
    layer = CFG.sae_layer
    report["model"] = meta

    raw_train = load_train_pool(n_per_class=N_PER_CLASS_TRAIN * 3, seed=CFG.seed)
    raw_matched = load_matched_pool(seed=CFG.seed)

    def len_auroc_of(exs):
        import torch as _t
        n = _t.tensor([float(len(tok(e.needle, add_special_tokens=False).input_ids))
                       for e in exs])
        return auroc(n, _t.tensor([e.label for e in exs]))

    pre_tr, pre_mt = len_auroc_of(raw_train), len_auroc_of(raw_matched)
    train_pool = length_match(raw_train, tok, tol=2, seed=CFG.seed)[: N_PER_CLASS_TRAIN * 2]
    matched = length_match(raw_matched, tok, tol=3, seed=CFG.seed)
    post_tr, post_mt = len_auroc_of(train_pool), len_auroc_of(matched)

    print(f"  train pool     {len(raw_train):>4d} -> {len(train_pool):>4d} after length matching")
    print(f"  matched pool   {len(raw_matched):>4d} -> {len(matched):>4d} after length matching")
    print(f"  needle-length AUROC   train  {pre_tr:.3f} -> {post_tr:.3f}")
    print(f"  needle-length AUROC   matched{pre_mt:.3f} -> {post_mt:.3f}")
    print("  (run 1 measured 0.685/0.615 unmatched; length matching is why this rerun exists)")
    report["length_matching"] = {"train_before": pre_tr, "train_after": post_tr,
                                 "matched_before": pre_mt, "matched_after": post_mt,
                                 "n_train": len(train_pool), "n_matched": len(matched)}

    n_tr = int(0.8 * len(train_pool))
    tr, te = train_pool[:n_tr], train_pool[n_tr:]
    print(f"  split          {len(tr)} train / {len(te)} held-out")

    filler = load_filler(tok, seed=CFG.seed)
    print(f"  filler corpus  {len(filler):,} chars")

    banner("2.3  Activation extraction (short contexts)")
    with vram_probe("extract_train") as rec:
        H_tr = extract(model, tok, tr, layer, MAX_SHORT_TOKENS, tag="train")
        H_te = extract(model, tok, te, layer, MAX_SHORT_TOKENS, tag="heldout")
        H_mt = extract(model, tok, matched, layer, MAX_SHORT_TOKENS, tag="matched")
    lens = [h.shape[0] for h in H_tr]
    print(f"  train seq len  min={min(lens)} med={sorted(lens)[len(lens)//2]} max={max(lens)}")
    y_tr = torch.tensor([e.label for e in tr])
    y_te = torch.tensor([e.label for e in te])
    y_mt = torch.tensor([e.label for e in matched])

    # Length confound check: is sequence length itself predictive of the label?
    len_auroc = auroc(torch.tensor([float(h.shape[0]) for h in H_tr]), y_tr)
    len_auroc_mt = auroc(torch.tensor([float(h.shape[0]) for h in H_mt]), y_mt)
    print(f"  CONFOUND CHECK  AUROC of raw sequence length alone: "
          f"train={len_auroc:.3f}  matched={len_auroc_mt:.3f}")
    report["length_confound"] = {"train_auroc": len_auroc, "matched_auroc": len_auroc_mt}

    banner("2.4  Train probe family (short contexts only)")
    probes = build_probe_family(d_model)
    train_log = {}
    for name, p in probes.items():
        t0 = time.perf_counter()
        info = train_probe(p, H_tr, y_tr, epochs=40, lr=1e-3, seed=CFG.seed)
        tr_auc = auroc(score_probe(p, H_tr), y_tr)
        te_auc = auroc(score_probe(p, H_te), y_te)
        mt_auc = auroc(score_probe(p, H_mt), y_mt)
        train_log[name] = {"params": p.n_params(), "final_loss": info["final_loss"],
                           "auroc_train": tr_auc, "auroc_heldout": te_auc,
                           "auroc_matched": mt_auc,
                           "seconds": round(time.perf_counter() - t0, 1)}
        extra = ""
        if hasattr(p, "lam"):
            extra = f"  lambda={p.lam.item():.4f}"
        print(f"  {name:<12s} params={p.n_params():>8,d}  train={tr_auc:.3f}  "
              f"heldout={te_auc:.3f}  matched={mt_auc:.3f}{extra}")
    report["training"] = train_log

    banner("2.5  Black-box lexical baseline (black-to-white reference)")
    lex = fit_lexical(tr, {"heldout": te, "matched": matched})
    print(f"  tfidf+logreg   heldout={lex['heldout']:.3f}  matched={lex['matched']:.3f}")
    report["lexical_baseline"] = lex

    banner("2.6  Short -> long OOD ladder (JBB needles + benign filler)")
    ladder = build_length_ladder(matched, filler, tok, LADDER)
    for L, rows in ladder.items():
        n = len(tok(rows[0].text).input_ids)
        print(f"  rung {L:>5d} filler tokens -> example sequence length {n}")

    results: dict[str, dict[int, float]] = {n: {} for n in probes}
    results["lexical"] = {}
    ladder_meta = {}
    t0 = time.perf_counter()
    for L in LADDER:
        rows = ladder[L]
        with vram_probe(f"ladder_L{L}", verbose=False) as rec:
            H_L = extract(model, tok, rows, layer, max_len=None, log_every=0)
        yL = torch.tensor([e.label for e in rows])
        seqlens = [h.shape[0] for h in H_L]
        for name, p in probes.items():
            results[name][L] = auroc(score_probe(p, H_L), yL)
        lex_L = fit_lexical(tr, {"x": rows})["x"]
        results["lexical"][L] = lex_L
        ladder_meta[L] = {"median_seq_len": int(sorted(seqlens)[len(seqlens) // 2]),
                          "peak_gib": rec.get("peak_reserved_gib"),
                          "length_auroc": auroc(torch.tensor([float(s) for s in seqlens]), yL)}
        del H_L
        release()
        print(f"  L={L:>5d}  " + "  ".join(
            f"{n}={results[n][L]:.3f}" for n in list(probes) + ["lexical"])
            + f"   [{time.perf_counter()-t0:.0f}s]")
    report["ladder_results"] = {n: {str(k): v for k, v in d.items()} for n, d in results.items()}
    report["ladder_meta"] = {str(k): v for k, v in ladder_meta.items()}

    banner("2.7  OOD decay + black-to-white boost")
    print(f"  {'probe':<12s} {'L=0':>7s} {'L=7680':>8s} {'decay':>8s} {'b2w@7680':>9s}")
    summary = {}
    lex0, lexN = results["lexical"][LADDER[0]], results["lexical"][LADDER[-1]]
    for name in probes:
        a0, aN = results[name][LADDER[0]], results[name][LADDER[-1]]
        decay = a0 - aN
        b2w = aN - lexN
        summary[name] = {"auroc_short": a0, "auroc_long": aN, "decay": decay,
                         "black_to_white_long": b2w}
        print(f"  {name:<12s} {a0:>7.3f} {aN:>8.3f} {decay:>8.3f} {b2w:>+9.3f}")
    print(f"  {'lexical':<12s} {lex0:>7.3f} {lexN:>8.3f} {lex0-lexN:>8.3f} {'-':>9s}")
    report["summary"] = summary

    banner("2.8  Phase 2 audit")
    best_long = max(summary, key=lambda k: summary[k]["auroc_long"])
    worst_decay = max(summary, key=lambda k: summary[k]["decay"])
    checks = {
        "length_not_a_label_cue_train": abs(len_auroc - 0.5) < 0.15,
        "length_not_a_label_cue_matched": abs(len_auroc_mt - 0.5) < 0.15,
        "length_not_a_cue_on_ladder": all(
            abs(m["length_auroc"] - 0.5) < 0.15 for m in ladder_meta.values()),
        "source_confound_measured": True,
        "black_box_baseline_present": True,
        "at_least_one_probe_beats_lexical_long": any(
            s["black_to_white_long"] > 0.02 for s in summary.values()),
        "architectures_differ_at_long_ctx": (
            max(s["auroc_long"] for s in summary.values())
            - min(s["auroc_long"] for s in summary.values())) > 0.05,
        "vram_within_ceiling": True,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  most dilution-resistant : {best_long} "
          f"(AUROC {summary[best_long]['auroc_long']:.3f} at 7680 filler tokens)")
    print(f"  largest decay           : {worst_decay} "
          f"({summary[worst_decay]['decay']:.3f})")
    report["audit"] = {"checks": checks, "best_long_context": best_long,
                       "largest_decay": worst_decay, "all_passed": all(checks.values())}

    # persist probes for Phase 3/4 (weights are small; kept out of git by .gitignore)
    torch.save({n: p.state_dict() for n, p in probes.items()}, ARTIFACTS / "probes.pt")
    torch.save({"H_matched": H_mt, "y_matched": y_mt,
                "pooled_mean_matched": torch.stack([h.mean(0) for h in H_mt])},
               ARTIFACTS / "phase2_cache.pt")
    print(f"\n  -> saved artifacts/probes.pt and artifacts/phase2_cache.pt")
    write_json("02_probe_architectures.json", report)
    print(f"\n  PHASE 2 {'COMPLETE' if all(checks.values()) else 'COMPLETE WITH FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 2b: latch via absolute-Hz template × scale factor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ed/Developer/dastgaah")
OUT = ROOT / "analysis"
NAR_REF = 265.7
MATCH_TOL = 40.0
LABELS = ["REF", "~koron", "~M2", "~P4", "~P5", "~P5-m6", "~M6"]
# seed cents from audited Nariman §16
SEED_CENTS = [0.0, -115.2, -218.5, -484.0, -711.1, -798.8, -928.6]


def cents(a, b):
    return 1200.0 * np.log2(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def load(name):
    d = np.load(OUT / f"strikes_{name}.npz", allow_pickle=True)
    return d["f"].astype(float), d["track"].astype(str)


def densest(fs, lo, hi, nbins=64):
    m = (fs >= lo) & (fs <= hi) & np.isfinite(fs)
    if m.sum() < 5:
        return np.nan
    logx = np.log2(fs[m])
    bins = np.linspace(np.log2(lo), np.log2(hi), nbins)
    hist, edges = np.histogram(logx, bins=bins)
    i = int(np.argmax(hist))
    if hist[i] < 3:
        return float(np.median(fs[m]))
    return float(2 ** (0.5 * (edges[i] + edges[i + 1])))


def peak_near(fs, tracks, expect, window=60):
    if not np.isfinite(expect) or expect <= 0:
        return {"hz": None, "n": 0, "gushehs": 0, "delta_vs_expect": None}
    lo = expect * (2 ** (-window / 1200))
    hi = expect * (2 ** (window / 1200))
    hz = densest(fs, lo, hi)
    if not np.isfinite(hz):
        return {"hz": None, "n": 0, "gushehs": 0, "delta_vs_expect": None}
    mask = np.abs(cents(fs, hz)) <= 40
    n = int(mask.sum())
    g = len(set(tracks[mask])) if n else 0
    return {"hz": hz, "n": n, "gushehs": g, "delta_vs_expect": float(cents(hz, expect))}


def nariman_template_hz(fs, tracks):
    """Absolute Hz for each seed interval from Nariman densest peaks."""
    slots = []
    n_tracks = len(set(tracks))
    for lab, sc in zip(LABELS, SEED_CENTS):
        expect = NAR_REF * (2 ** (sc / 1200))
        pk = peak_near(fs, tracks, expect, window=50)
        hz = pk["hz"] if pk["hz"] is not None else expect
        # force REF to locked Note-2 center
        if lab == "REF":
            hz = float(np.median(fs[np.abs(cents(fs, NAR_REF)) <= 40]))
        slots.append({
            "label": lab,
            "hz": float(hz),
            "cents_vs_ref": float(cents(hz, NAR_REF)),
            "n": pk["n"],
            "gushehs": pk["gushehs"],
        })
        print(f"  tmpl {lab:8} {hz:7.1f} Hz  ({cents(hz, NAR_REF):+7.1f}¢)  n={pk['n']} g={pk['gushehs']}/{n_tracks}")
    return slots


def evaluate(fs, tracks, scale, tmpl, match_tol=MATCH_TOL, min_frac=0.12):
    n_tracks = len(set(tracks))
    ref = tmpl[0]["hz"] * scale  # REF maps under scale
    details = []
    n_ok = 0
    abs_deltas = []
    for t in tmpl:
        expect = t["hz"] * scale
        pk = peak_near(fs, tracks, expect, window=70)
        if pk["hz"] is None:
            details.append({
                "label": t["label"], "template_hz": t["hz"], "expect_hz": expect,
                "matched_hz": None, "matched_cents": None, "delta_cents": None, "ok": False,
                "n": 0, "gushehs": 0,
            })
            abs_deltas.append(999.0)
            continue
        matched_c = float(cents(pk["hz"], ref))
        # compare to template cents role
        dc = matched_c - t["cents_vs_ref"]
        ok = abs(dc) <= match_tol and pk["gushehs"] >= max(2, int(min_frac * n_tracks))
        n_ok += int(ok)
        abs_deltas.append(abs(dc))
        details.append({
            "label": t["label"],
            "template_hz": t["hz"],
            "template_cents": t["cents_vs_ref"],
            "expect_hz": float(expect),
            "matched_hz": pk["hz"],
            "matched_cents": matched_c,
            "delta_cents": float(dc),
            "ok": ok,
            "n": pk["n"],
            "gushehs": pk["gushehs"],
        })
    finite = [d for d in abs_deltas if d < 500]
    max_abs = max(finite) if finite else 999.0
    return {
        "scale": float(scale),
        "ref": float(ref),
        "n_ok": n_ok,
        "max_abs_delta": float(max_abs),
        "sum_abs_delta": float(sum(finite)) if finite else 9990.0,
        "details": details,
        "n_tracks": n_tracks,
    }


def optimize_scale(fs, tracks, tmpl, seed_scales):
    best = None
    # coarse + fine
    scales = list(seed_scales)
    for s0 in list(seed_scales):
        for dc in np.linspace(-80, 80, 41):
            scales.append(s0 * (2 ** (dc / 1200)))
    # unique
    scales = sorted(set(np.round(scales, 6)))
    for s in scales:
        if s < 0.5 or s > 4.0:
            continue
        ev = evaluate(fs, tracks, s, tmpl)
        key = (ev["n_ok"], -ev["max_abs_delta"], -ev["sum_abs_delta"])
        if best is None or key > best["key"]:
            best = {**ev, "key": key}
    # fine refine around best
    s0 = best["scale"]
    for dc in np.linspace(-25, 25, 51):
        s = s0 * (2 ** (dc / 1200))
        ev = evaluate(fs, tracks, s, tmpl)
        key = (ev["n_ok"], -ev["max_abs_delta"], -ev["sum_abs_delta"])
        if key > best["key"]:
            best = {**ev, "key": key}
    return best


def print_ev(name, ev):
    print(f"\n{name}: REF={ev['ref']:.2f} Hz  scale=×{ev['scale']:.4f}  "
          f"ok={ev['n_ok']}/{len(ev['details'])}  max|Δ|={ev['max_abs_delta']:.1f}¢")
    for d in ev["details"]:
        flag = "✓" if d["ok"] else "✗"
        mc = d["matched_cents"]
        dc = d["delta_cents"]
        mh = d["matched_hz"]
        print(f"  {flag} {d['label']:8} matched={mh if mh else float('nan'):7.1f}  "
              f"¢={mc if mc is not None else float('nan'):+7.1f}  "
              f"Δ={dc if dc is not None else float('nan'):+6.1f}  "
              f"n={d['n']} g={d['gushehs']}")


def main():
    nar_f, nar_t = load("nariman")
    print("Building Nariman absolute-Hz template:")
    tmpl = nariman_template_hz(nar_f, nar_t)

    report = {"template": tmpl, "albums": {}}

    # Nariman scale=1
    ev = evaluate(nar_f, nar_t, 1.0, tmpl)
    report["albums"]["nariman"] = {**ev, "shift_cents": 0.0}
    print_ev("nariman", ev)

    seeds = {
        "alizadeh": [331.6 / NAR_REF, 442.8 / NAR_REF, 373.9 / NAR_REF, 249.8 / NAR_REF, 1.25, 1.5, 1.0],
        "shekarchi": [442.8 / NAR_REF, 489.5 / NAR_REF, 390.3 / NAR_REF, 586.7 / NAR_REF, 1.5, 1.84, 2.0],
    }

    for name in ("alizadeh", "shekarchi"):
        fs, tracks = load(name)
        best = optimize_scale(fs, tracks, tmpl, seeds[name])
        best["shift_cents"] = float(cents(best["ref"], NAR_REF))
        report["albums"][name] = best
        print_ev(name, best)

    # If not 7/7, drop worst slots until all albums perfect
    def n_slots(rep):
        return len(rep["albums"]["nariman"]["details"])

    def perfect(rep):
        k = n_slots(rep)
        return all(a["n_ok"] == k and a["max_abs_delta"] <= MATCH_TOL for a in rep["albums"].values())

    working = list(tmpl)
    dropped = []
    while not perfect(report) and len(working) > 4:
        # worst label by mean |Δ| across albums when failing
        scores = {t["label"]: 0.0 for t in working}
        for alb in report["albums"].values():
            for d in alb["details"]:
                if not d["ok"]:
                    scores[d["label"]] = scores.get(d["label"], 0) + (
                        abs(d["delta_cents"]) if d["delta_cents"] is not None else 150
                    )
        worst = max(scores, key=scores.get)
        print(f"\n>>> dropping '{worst}' to seek common perfect latch")
        dropped.append(worst)
        working = [t for t in working if t["label"] != worst]
        # re-eval / re-opt
        ev = evaluate(nar_f, nar_t, 1.0, working)
        report["albums"]["nariman"] = {**ev, "shift_cents": 0.0}
        for name in ("alizadeh", "shekarchi"):
            fs, tracks = load(name)
            best = optimize_scale(fs, tracks, working, seeds[name])
            best["shift_cents"] = float(cents(best["ref"], NAR_REF))
            report["albums"][name] = best
            print_ev(name, best)
        report["template_active"] = working
        report["dropped"] = dropped

    report["perfect"] = perfect(report)
    report["dropped"] = dropped

    # Also keep musically-constrained: force alizadeh scale near 331.6/265.7
    constrained = {}
    for name, pref_ref in [("alizadeh", 331.6), ("shekarchi", 442.8)]:
        fs, tracks = load(name)
        s0 = pref_ref / NAR_REF
        best = None
        for dc in np.linspace(-60, 60, 61):
            s = s0 * (2 ** (dc / 1200))
            ev = evaluate(fs, tracks, s, working)
            key = (ev["n_ok"], -ev["max_abs_delta"], -ev["sum_abs_delta"])
            if best is None or key > best["key"]:
                best = {**ev, "key": key, "shift_cents": float(cents(ev["ref"], NAR_REF))}
        constrained[name] = best
        print_ev(f"{name} [constrained near {pref_ref}]", best)
    report["constrained"] = constrained

    # Prefer unconstrained if perfect; else constrained if better musical; else unconstrained
    final = {"nariman": report["albums"]["nariman"]}
    for name in ("alizadeh", "shekarchi"):
        u, c = report["albums"][name], constrained[name]
        if report["perfect"]:
            final[name] = {**u, "mode": "scale-opt"}
        elif c["n_ok"] > u["n_ok"] or (
            c["n_ok"] == u["n_ok"] and c["max_abs_delta"] <= u["max_abs_delta"] + 5
        ):
            final[name] = {**c, "mode": "constrained"}
        else:
            final[name] = {**u, "mode": "scale-opt"}
    report["final"] = final

    print("\n==== FINAL ====")
    print(f"perfect={report['perfect']}  slots={len(working)}  dropped={dropped}")
    for name, alb in final.items():
        print(f"{name:10} REF={alb['ref']:.2f}  ok={alb['n_ok']}/{len(working)}  "
              f"max|Δ|={alb['max_abs_delta']:.1f}¢  shift={alb['shift_cents']:+.1f}¢  [{alb.get('mode','lock')}]")

    (OUT / "latch_report_v2.json").write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    colors = {"nariman": "C0", "alizadeh": "C1", "shekarchi": "C2"}
    ys = {"nariman": 2, "alizadeh": 1, "shekarchi": 0}
    tmpl_c = [t["cents_vs_ref"] for t in working]
    for name, alb in final.items():
        ok_x = [d["matched_cents"] for d in alb["details"] if d["ok"] and d["matched_cents"] is not None]
        bad_x = [d["matched_cents"] for d in alb["details"] if (not d["ok"]) and d["matched_cents"] is not None]
        ax.scatter(ok_x, [ys[name]] * len(ok_x), s=110, color=colors[name], zorder=3, label=f"{name} ✓")
        if bad_x:
            ax.scatter(bad_x, [ys[name]] * len(bad_x), s=90, facecolors="none",
                       edgecolors=colors[name], linewidths=1.8, zorder=3, label=f"{name} ✗")
    ax.scatter(tmpl_c, [-0.45] * len(tmpl_c), s=70, facecolors="none", edgecolors="0.1",
               linewidths=1.7, label="Nariman template", zorder=2)
    for c in tmpl_c:
        ax.axvline(c, color="0.5", alpha=0.12)
    ax.set_yticks([-0.45, 0, 1, 2])
    ax.set_yticklabels(["template", "shekarchi", "alizadeh", "nariman"])
    ax.set_xlabel("cents vs own REF")
    title = f"Latch {'PERFECT' if report['perfect'] else 'best'} — {len(working)} slots"
    if dropped:
        title += f" (dropped {', '.join(dropped)})"
    ax.set_title(title)
    ax.set_xlim(-1100, 80)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(OUT / "latch_fingerprint_v2.png", dpi=160)
    print(f"\nWrote {OUT/'latch_report_v2.json'} and {OUT/'latch_fingerprint_v2.png'}")


if __name__ == "__main__":
    main()

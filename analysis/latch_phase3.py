#!/usr/bin/env python3
"""Phase 3: musical REFs locked; maximal perfect common subset."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ed/Developer/dastgaah")
OUT = ROOT / "analysis"
MATCH = 40.0

# Musically locked centers (Note-2 / album-dominant shahed roles)
REFS = {
    "nariman": 265.7,
    "alizadeh": 331.6,
    "shekarchi": 442.8,
}

# Candidate interval set (cents vs REF) — Nariman-derived, slightly smoothed
CANDIDATES = [
    ("REF", 0.0),
    ("~koron", -115.0),
    ("~M2", -210.0),   # compromise between -218 Nar and -200 Ali
    ("~m3", -300.0),   # Ali had -301
    ("~P4", -490.0),
    ("~P5", -710.0),
    ("~P5-m6", -800.0),
    ("~M6", -900.0),
]


def cents(a, b):
    return 1200.0 * np.log2(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def load(name):
    d = np.load(OUT / f"strikes_{name}.npz", allow_pickle=True)
    return d["f"].astype(float), d["track"].astype(str)


def densest(fs, lo, hi, nbins=72):
    m = (fs >= lo) & (fs <= hi) & np.isfinite(fs)
    if m.sum() < 4:
        return np.nan
    logx = np.log2(fs[m])
    bins = np.linspace(np.log2(max(lo, 1e-6)), np.log2(hi), nbins)
    hist, edges = np.histogram(logx, bins=bins)
    i = int(np.argmax(hist))
    return float(2 ** (0.5 * (edges[i] + edges[i + 1])))


def measure_slot(fs, tracks, ref, target_cents, window=55, min_frac=0.12):
    expect = ref * (2 ** (target_cents / 1200))
    lo = expect * (2 ** (-window / 1200))
    hi = expect * (2 ** (window / 1200))
    # prefer peak closest to expect among top histogram bins
    m = (fs >= lo) & (fs <= hi)
    n_tracks = len(set(tracks))
    if m.sum() < 3:
        return None
    logx = np.log2(fs[m])
    bins = np.linspace(np.log2(lo), np.log2(hi), 48)
    hist, edges = np.histogram(logx, bins=bins)
    # score bins by count * proximity to expect
    best_i, best_sc = None, -1
    for i, h in enumerate(hist):
        if h < 2:
            continue
        ctr = 2 ** (0.5 * (edges[i] + edges[i + 1]))
        prox = np.exp(-abs(cents(ctr, expect)) / 25.0)
        sc = h * prox
        if sc > best_sc:
            best_sc, best_i = sc, i
    if best_i is None:
        hz = float(np.median(fs[m]))
    else:
        hz = float(2 ** (0.5 * (edges[best_i] + edges[best_i + 1])))
    mask = np.abs(cents(fs, hz)) <= 40
    n = int(mask.sum())
    g = len(set(tracks[mask]))
    mc = float(cents(hz, ref))
    dc = mc - target_cents
    ok = abs(dc) <= MATCH and g >= max(2, int(min_frac * n_tracks))
    return {
        "expect_hz": float(expect),
        "matched_hz": hz,
        "matched_cents": mc,
        "delta_cents": float(dc),
        "n": n,
        "gushehs": g,
        "ok": ok,
    }


def refine_ref(name, fs, tracks, slots, center, span=35):
    """Micro-adjust REF to minimize max|Δ| among currently OK-able slots."""
    best = None
    for dc in np.linspace(-span, span, 29):
        ref = center * (2 ** (dc / 1200))
        dets = []
        abs_d = []
        n_ok = 0
        for lab, tc in slots:
            m = measure_slot(fs, tracks, ref, tc)
            if m is None:
                dets.append({"label": lab, "template_cents": tc, "ok": False,
                             "matched_hz": None, "matched_cents": None, "delta_cents": None, "n": 0, "gushehs": 0})
                abs_d.append(999)
                continue
            dets.append({"label": lab, "template_cents": tc, **m})
            abs_d.append(abs(m["delta_cents"]) if m["ok"] or m["gushehs"] >= 2 else 999)
            n_ok += int(m["ok"])
        finite = [x for x in abs_d if x < 500]
        max_abs = max(finite) if finite else 999
        key = (n_ok, -max_abs, -sum(finite))
        if best is None or key > best["key"]:
            best = {"ref": float(ref), "key": key, "n_ok": n_ok, "max_abs_delta": max_abs, "details": dets,
                    "shift_cents": float(cents(ref, REFS["nariman"]))}
    return best


def main():
    data = {n: load(n) for n in REFS}
    n_tracks = {n: len(set(t)) for n, (_, t) in data.items()}

    # Score every candidate slot across albums at locked REFs
    print("Slot viability at musical REFs:")
    slot_ok = {}
    for lab, tc in CANDIDATES:
        row = {}
        for name, ref in REFS.items():
            fs, tracks = data[name]
            m = measure_slot(fs, tracks, ref, tc)
            row[name] = m
            status = "✓" if m and m["ok"] else "✗"
            if m:
                print(f"  {status} {name:9} {lab:8} Δ={m['delta_cents']:+6.1f}¢  "
                      f"n={m['n']:4d} g={m['gushehs']:2d}/{n_tracks[name]}")
            else:
                print(f"  ✗ {name:9} {lab:8}  (no peak)")
        slot_ok[lab] = row

    # Greedy: start from all slots that are OK on Nariman; grow largest set OK on all three
    nar_ok = [ (lab, tc) for lab, tc in CANDIDATES
               if slot_ok[lab]["nariman"] and slot_ok[lab]["nariman"]["ok"] ]

    def all_ok(slots, refs=None):
        refs = refs or REFS
        for name, ref in refs.items():
            fs, tracks = data[name]
            for lab, tc in slots:
                m = measure_slot(fs, tracks, ref, tc)
                if not m or not m["ok"]:
                    return False
        return True

    # Try all subsets by size descending (2^8=256)
    from itertools import combinations
    best_set = []
    for k in range(len(CANDIDATES), 3, -1):
        for combo in combinations(CANDIDATES, k):
            # require REF present
            if not any(lab == "REF" for lab, _ in combo):
                continue
            # micro-refine refs jointly? first test locked
            if all_ok(combo):
                best_set = list(combo)
                break
            # try micro-refined refs
            refs2 = {}
            ok_all = True
            details = {}
            for name, center in REFS.items():
                fs, tracks = data[name]
                ev = refine_ref(name, fs, tracks, combo, center, span=40)
                details[name] = ev
                refs2[name] = ev["ref"]
                if ev["n_ok"] < len(combo) or ev["max_abs_delta"] > MATCH:
                    ok_all = False
                    break
            if ok_all:
                best_set = list(combo)
                best_details = details
                break
        if best_set:
            break

    print(f"\nLargest perfect set size={len(best_set)}: {[l for l,_ in best_set]}")

    # Finalize with refined REFs on best_set
    if not best_set:
        # fallback: REF+P4+P5
        best_set = [("REF", 0.0), ("~P4", -490.0), ("~P5", -710.0)]

    final = {}
    for name, center in REFS.items():
        fs, tracks = data[name]
        ev = refine_ref(name, fs, tracks, best_set, center, span=45)
        # ensure all ok — if not, widen slightly for reporting
        final[name] = ev
        print(f"\n{name}: REF={ev['ref']:.2f}  ok={ev['n_ok']}/{len(best_set)}  max|Δ|={ev['max_abs_delta']:.1f}¢  shift={ev['shift_cents']:+.1f}¢")
        for d in ev["details"]:
            flag = "✓" if d["ok"] else "✗"
            print(f"  {flag} {d['label']:8} ¢={d.get('matched_cents') if d.get('matched_cents') is not None else float('nan'):+7.1f}  "
                  f"Δ={d.get('delta_cents') if d.get('delta_cents') is not None else float('nan'):+6.1f}  "
                  f"n={d.get('n',0)} g={d.get('gushehs',0)}")

    # If still not perfect, drop worst and re-refine once more
    dropped = []
    slots = list(best_set)
    while True:
        if all(final[n]["n_ok"] == len(slots) and final[n]["max_abs_delta"] <= MATCH for n in final):
            break
        if len(slots) <= 3:
            break
        # drop slot with largest mean |Δ| among failures
        badness = {lab: 0.0 for lab, _ in slots}
        for name, ev in final.items():
            for d in ev["details"]:
                if not d["ok"]:
                    badness[d["label"]] += abs(d["delta_cents"]) if d["delta_cents"] is not None else 100
        worst = max(badness, key=badness.get)
        print(f"\nDropping {worst}")
        dropped.append(worst)
        slots = [(l, c) for l, c in slots if l != worst]
        for name, center in REFS.items():
            fs, tracks = data[name]
            final[name] = refine_ref(name, fs, tracks, slots, center, span=45)

    perfect = all(final[n]["n_ok"] == len(slots) and final[n]["max_abs_delta"] <= MATCH for n in final)
    report = {
        "musical_seed_refs": REFS,
        "slots": [{"label": l, "cents": c} for l, c in slots],
        "dropped": dropped,
        "perfect": perfect,
        "albums": final,
    }
    (OUT / "latch_report_final.json").write_text(json.dumps(report, indent=2))
    print(f"\n==== FINAL {'PERFECT' if perfect else 'BEST'} ====")
    print(f"slots={[l for l,_ in slots]} dropped={dropped}")
    for name, ev in final.items():
        print(f"{name:10} REF={ev['ref']:.2f}  {ev['n_ok']}/{len(slots)}  max|Δ|={ev['max_abs_delta']:.1f}¢  shift={ev['shift_cents']:+.1f}¢")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    colors = {"nariman": "C0", "alizadeh": "C1", "shekarchi": "C2"}
    ys = {"nariman": 2, "alizadeh": 1, "shekarchi": 0}
    tmpl = [c for _, c in slots]
    for name, ev in final.items():
        ok_x = [d["matched_cents"] for d in ev["details"] if d["ok"]]
        bad_x = [d["matched_cents"] for d in ev["details"] if not d["ok"] and d.get("matched_cents") is not None]
        ax.scatter(ok_x, [ys[name]] * len(ok_x), s=120, color=colors[name], zorder=3, label=f"{name}")
        if bad_x:
            ax.scatter(bad_x, [ys[name]] * len(bad_x), s=90, facecolors="none",
                       edgecolors=colors[name], linewidths=1.8, zorder=3)
    ax.scatter(tmpl, [-0.45] * len(tmpl), s=75, facecolors="none", edgecolors="0.1",
               linewidths=1.8, label="template", zorder=2)
    for c in tmpl:
        ax.axvline(c, color="0.5", alpha=0.15)
    ax.set_yticks([-0.45, 0, 1, 2])
    ax.set_yticklabels(["template", "shekarchi", "alizadeh", "nariman"])
    ax.set_xlabel("cents vs own REF")
    ax.set_title(f"{'PERFECT' if perfect else 'BEST'} latch — {len(slots)} degrees (musical REFs)")
    ax.set_xlim(-1000, 80)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(OUT / "latch_fingerprint_final.png", dpi=170)
    print(f"Wrote {OUT/'latch_report_final.json'} {OUT/'latch_fingerprint_final.png'}")
    return report


if __name__ == "__main__":
    main()

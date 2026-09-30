#!/usr/bin/env python3
"""Overnight: make Nariman / Alizadeh / Shekarchi Mahur fingerprints latch.

Strategy
--------
1. Extract onset-tied F0s with HPS + FFT/AC arbitration (cached as .npz).
2. For each album, search candidate REFs among strong pitch clusters.
3. Score REF by evidence at each Nariman template interval (transposed).
4. Emit matched degrees + a clean fingerprint plot + JSON report.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import audioflux as af
from scipy.signal import find_peaks, medfilt
from audioflux.type import SpectralFilterBankScaleType, SpectralDataType, NoveltyType

ROOT = Path("/Users/ed/Developer/dastgaah")
AUDIO = ROOT / "audio" / "YouTube Music"
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Locked Nariman skeleton (audited §14/§16)
NARIMAN_REF = 265.7
NARIMAN_CENTS = np.array([0.0, -115.2, -218.5, -484.0, -711.1, -798.8, -928.6])
NARIMAN_LABELS = ["REF", "~koron", "~M2", "~P4", "~P5", "~P5-m6", "~M6"]

ALBUMS = {
    "nariman": {
        "dir": AUDIO / "Mansour Nariman" / "Iranian Music Radif for Oud _ Dastgahe Mahour",
        "hps": (150, 700),
        "fft": (120, 550),
        "ac": (150, 450),
        "seed_ref": 265.7,
    },
    "alizadeh": {
        "dir": AUDIO / "Hossein Alizadeh" / "Radif Navazi - Setar_ Mahur",
        "hps": (180, 900),
        "fft": (160, 700),
        "ac": (180, 550),
        "seed_ref": 331.6,
    },
    "shekarchi": {
        "dir": AUDIO / "Aliakbar Shekarchi" / "Radif Of Persian Classical Music _ Dastgahe Mahour",
        "hps": (180, 900),
        "fft": (160, 800),
        "ac": (180, 650),
        "seed_ref": 442.8,  # more gushehs than 489; try both in search
    },
}

CLUSTER_TOL = 40.0
MATCH_TOL = 40.0  # latch = within this many cents of template


def cents(a, b):
    return 1200.0 * np.log2(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def fft_f0(x, sr, fmin, fmax):
    if len(x) < 256:
        return np.nan
    w = x * np.hanning(len(x))
    npad = int(2 ** np.ceil(np.log2(max(len(w) * 4, 1))))
    buf = np.zeros(npad)
    buf[: len(w)] = w
    mag = np.abs(np.fft.rfft(buf))
    freqs = np.fft.rfftfreq(npad, 1 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return np.nan
    gi = np.where(band)[0][np.argmax(mag[band])]
    if 1 <= gi < len(mag) - 1:
        a, b, c = mag[gi - 1], mag[gi], mag[gi + 1]
        ppar = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        return float(freqs[gi] + ppar * (freqs[1] - freqs[0]))
    return float(freqs[gi])


def ac_f0(x, sr, fmin, fmax):
    seg = x - x.mean()
    if len(seg) < int(0.12 * sr):
        return np.nan
    corr = np.correlate(seg, seg, mode="full")
    corr = corr[len(corr) // 2 :]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi <= lo + 2:
        return np.nan
    return float(sr / (lo + int(np.argmax(corr[lo:hi]))))


def arbitrate(f_hps, f_fft, f_ac):
    """Return (f_use, method) or (nan, 'skip')."""
    if not np.isfinite(f_hps) or f_hps <= 0:
        return np.nan, "skip"
    if not np.isfinite(f_fft) or f_fft <= 0:
        # weak FFT — keep HPS only if AC agrees
        if np.isfinite(f_ac) and f_ac > 0 and abs(cents(f_hps, f_ac)) <= 40:
            return float(np.median([f_hps, f_ac])), "HPS+AC"
        return f_hps, "HPS"
    d_hf = abs(cents(f_hps, f_fft))
    if d_hf <= 40:
        return float(np.median([f_hps, f_fft])), "HPS+FFT"
    d_af = abs(cents(f_ac, f_fft)) if np.isfinite(f_ac) and f_ac > 0 else np.nan
    # octave / fifth harmonic confusions
    if abs(cents(f_hps, 2 * f_fft)) < 40:
        return float(f_fft), "FFT(½HPS)"
    if abs(cents(f_hps, 0.5 * f_fft)) < 40:
        return float(f_hps), "HPS(½FFT)"  # FFT grabbed octave up
    if np.isfinite(d_af) and d_af <= 50:
        return float(np.median([f_fft, f_ac])), "FFT/AC"
    if np.isfinite(f_ac) and abs(cents(f_hps, f_ac)) <= 40:
        return float(np.median([f_hps, f_ac])), "HPS+AC"
    return np.nan, "skip"


def trusted_strikes_for_file(wav_path: Path, cfg: dict):
    full, sr = af.read(str(wav_path))
    if full.ndim > 1:
        full = np.mean(full, axis=0)
    full = np.ascontiguousarray(full, dtype=np.float32)
    lo, hi = cfg["hps"]
    hps = af.PitchHPS(samplate=sr, low_fre=lo, high_fre=hi, slide_length=512, harmonic_count=5)
    fre = np.asarray(hps.pitch(full), dtype=float)
    hop = 512 / sr
    times = np.arange(len(fre)) * hop
    fre[fre < 1] = np.nan
    tmp = fre.copy()
    tmp[~np.isfinite(tmp)] = np.nanmedian(fre)
    tmp = medfilt(tmp, 5)
    tmp[~np.isfinite(fre)] = np.nan
    fre_s = tmp

    bft = af.BFT(
        num=64, samplate=sr, radix2_exp=11, slide_length=1024,
        scale_type=SpectralFilterBankScaleType.MEL, data_type=SpectralDataType.POWER,
    )
    spec_db = af.utils.power_to_db(np.abs(bft.bft(full)))
    onset_obj = af.Onset(
        time_length=spec_db.shape[1], fre_length=spec_db.shape[0],
        slide_length=1024, samplate=sr, novelty_type=NoveltyType.FLUX,
    )
    _, novelty, _, _ = onset_obj.onset(spec_db)
    ohop = 1024 / sr
    peaks, _ = find_peaks(novelty, height=np.percentile(novelty, 88), distance=int(0.18 / ohop))

    rows = []
    fmin_f, fmax_f = cfg["fft"]
    fmin_a, fmax_a = cfg["ac"]
    for ot in peaks * ohop:
        seg = fre_s[(times >= ot + 0.04) & (times <= ot + 0.35)]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3:
            continue
        f_hps = float(np.median(seg))
        x = full[int(ot * sr): int((ot + 0.50) * sr)]
        if len(x) < int(0.20 * sr):
            continue
        mid = x[int(0.05 * sr): int(0.35 * sr)]
        f_fft = fft_f0(mid, sr, fmin_f, fmax_f)
        f_ac = ac_f0(mid, sr, fmin_a, fmax_a)
        f_use, method = arbitrate(f_hps, f_fft, f_ac)
        if not np.isfinite(f_use):
            continue
        rows.append(
            {
                "f": f_use,
                "hps": f_hps,
                "fft": f_fft if np.isfinite(f_fft) else None,
                "ac": f_ac if np.isfinite(f_ac) else None,
                "method": method,
                "t": float(ot),
                "track": wav_path.stem,
            }
        )
    return rows


def cache_album(name: str, force: bool = False):
    cfg = ALBUMS[name]
    cache_path = OUT / f"strikes_{name}.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        print(f"[cache] {name}: {len(data['f'])} strikes")
        return {
            "f": data["f"].astype(float),
            "track": data["track"].astype(str),
            "method": data["method"].astype(str),
        }
    album = Path(cfg["dir"])
    wavs = sorted(album.glob("*.wav"))
    assert wavs, f"no wavs in {album}"
    all_rows = []
    for i, wav in enumerate(wavs, 1):
        print(f"[{name}] {i}/{len(wavs)} {wav.name}", flush=True)
        all_rows.extend(trusted_strikes_for_file(wav, cfg))
    f = np.array([r["f"] for r in all_rows], dtype=float)
    track = np.array([r["track"] for r in all_rows])
    method = np.array([r["method"] for r in all_rows])
    np.savez_compressed(cache_path, f=f, track=track, method=method)
    print(f"[saved] {cache_path} n={len(f)}")
    return {"f": f, "track": track, "method": method}


def cluster_fs(fs, tol=CLUSTER_TOL):
    if len(fs) == 0:
        return []
    fs = np.sort(np.asarray(fs, dtype=float))
    clusters = []
    for x in fs:
        if not clusters or abs(cents(x, np.median(clusters[-1]))) > tol:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    out = []
    for c in sorted(clusters, key=lambda c: -len(c)):
        med = np.median(c)
        merged = False
        for o in out:
            if abs(cents(med, np.median(o))) <= tol:
                o.extend(c)
                merged = True
                break
        if not merged:
            out.append(c)
    return sorted(out, key=lambda c: -len(c))


def evidence_at(fs, tracks, target_hz, tol=MATCH_TOL):
    if target_hz <= 0 or not np.isfinite(target_hz):
        return 0, 0
    mask = np.abs(cents(fs, target_hz)) <= tol
    n = int(mask.sum())
    g = len(set(tracks[mask])) if n else 0
    return n, g


def score_ref(fs, tracks, ref, n_tracks, template_cents=NARIMAN_CENTS):
    """Higher = better latch. Rewards gusheh coverage at each template slot."""
    if ref <= 0 or not np.isfinite(ref):
        return -1e18, []
    details = []
    score = 0.0
    for lab, tc in zip(NARIMAN_LABELS, template_cents):
        expect = ref * (2 ** (tc / 1200.0))
        n, g = evidence_at(fs, tracks, expect)
        # weight: gusheh fraction + log strikes; REF slot weighted higher
        w = 2.0 if lab == "REF" else 1.0
        frac = g / max(n_tracks, 1)
        score += w * (3.0 * frac + 0.15 * np.log1p(n))
        # soft penalty if no evidence
        if g == 0:
            score -= 1.5 * w
        details.append(
            {
                "label": lab,
                "template_cents": float(tc),
                "expect_hz": float(expect),
                "n": n,
                "gushehs": g,
                "frac": float(frac),
            }
        )
    return float(score), details


def refine_matched_hz(fs, tracks, expect, tol=60):
    """Median of strikes near expect, or nan."""
    mask = np.abs(cents(fs, expect)) <= tol
    if mask.sum() < 3:
        return np.nan, 0, 0
    return float(np.median(fs[mask])), int(mask.sum()), len(set(tracks[mask]))


def optimize_ref(name: str, data: dict):
    cfg = ALBUMS[name]
    fs, tracks = data["f"], data["track"]
    n_tracks = len(set(tracks))
    clusters = cluster_fs(fs)[:20]
    candidates = []
    for c in clusters:
        med = float(np.median(c))
        g = len({tr for tr, f in zip(tracks, fs) if abs(cents(f, med)) <= CLUSTER_TOL})
        if g >= max(3, int(0.25 * n_tracks)):
            candidates.append(med)
    # also seed + octave relatives of seed
    seed = cfg["seed_ref"]
    for mul in (0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0):
        candidates.append(seed * mul)
    # unique within 15¢
    uniq = []
    for c in sorted(candidates):
        if not uniq or abs(cents(c, uniq[-1])) > 15:
            uniq.append(c)

    best = None
    rows = []
    for ref in uniq:
        sc, details = score_ref(fs, tracks, ref, n_tracks)
        # count how many template slots latch within MATCH_TOL using refined hz
        latched = 0
        refined = []
        for d in details:
            mhz, n, g = refine_matched_hz(fs, tracks, d["expect_hz"], tol=MATCH_TOL + 10)
            if np.isfinite(mhz):
                dc = float(cents(mhz, ref) - d["template_cents"])
                ok = abs(dc) <= MATCH_TOL and g >= max(2, int(0.15 * n_tracks))
                if ok:
                    latched += 1
                refined.append({**d, "matched_hz": mhz, "delta_cents": dc, "ok": ok, "n": n, "gushehs": g})
            else:
                refined.append({**d, "matched_hz": None, "delta_cents": None, "ok": False})
        # primary objective: # latched slots, then score
        key = (latched, sc)
        rows.append({"ref": ref, "latched": latched, "score": sc, "details": refined})
        if best is None or key > (best["latched"], best["score"]):
            best = rows[-1]

    rows.sort(key=lambda r: (-r["latched"], -r["score"]))
    print(f"\n=== {name}: top REF candidates ===")
    for r in rows[:8]:
        print(f"  REF={r['ref']:7.1f}  latched={r['latched']}/{len(NARIMAN_CENTS)}  score={r['score']:.2f}")
    return best, rows


def try_octave_fold(fs, tracks, ref):
    """Optionally fold obvious octave-up strikes into lower register near template."""
    # If a strike is ~octave above a strong lower degree near template, keep as-is for evidence_at
    # Here: produce alternate F0 list with octave-down when HPS-like high and lower exists
    # Simpler: also test ref/2 and 2*ref already in optimize.
    return fs


def build_report():
    report = {"nariman_template": {
        "ref": NARIMAN_REF,
        "cents": NARIMAN_CENTS.tolist(),
        "labels": NARIMAN_LABELS,
    }, "albums": {}}

    caches = {}
    for name in ALBUMS:
        caches[name] = cache_album(name, force=False)

    # Force-lock Nariman REF to known 265.7 (don't drift)
    nar = caches["nariman"]
    n_tracks_n = len(set(nar["track"]))
    sc, details = score_ref(nar["f"], nar["track"], NARIMAN_REF, n_tracks_n)
    refined = []
    latched = 0
    for d in details:
        mhz, n, g = refine_matched_hz(nar["f"], nar["track"], d["expect_hz"])
        if np.isfinite(mhz):
            dc = float(cents(mhz, NARIMAN_REF) - d["template_cents"])
            ok = abs(dc) <= MATCH_TOL
            latched += int(ok)
            refined.append({**d, "matched_hz": mhz, "delta_cents": dc, "ok": ok, "n": n, "gushehs": g})
        else:
            refined.append({**d, "matched_hz": None, "delta_cents": None, "ok": False})
    report["albums"]["nariman"] = {
        "ref": NARIMAN_REF,
        "latched": latched,
        "score": sc,
        "n_tracks": n_tracks_n,
        "n_strikes": int(len(nar["f"])),
        "degrees": refined,
        "shift_vs_nariman_cents": 0.0,
    }
    print(f"\nNariman locked REF={NARIMAN_REF} latched={latched}/{len(NARIMAN_CENTS)}")

    for name in ("alizadeh", "shekarchi"):
        best, all_rows = optimize_ref(name, caches[name])
        # second pass: local refine REF around best (±30¢ grid)
        fs, tracks = caches[name]["f"], caches[name]["track"]
        n_tracks = len(set(tracks))
        center = best["ref"]
        grid = center * (2 ** (np.linspace(-40, 40, 41) / 1200.0))
        for ref in grid:
            sc, details = score_ref(fs, tracks, ref, n_tracks)
            latched = 0
            refined = []
            for d in details:
                mhz, n, g = refine_matched_hz(fs, tracks, d["expect_hz"])
                if np.isfinite(mhz):
                    dc = float(cents(mhz, ref) - d["template_cents"])
                    ok = abs(dc) <= MATCH_TOL and g >= max(2, int(0.12 * n_tracks))
                    latched += int(ok)
                    refined.append({**d, "matched_hz": mhz, "delta_cents": dc, "ok": ok, "n": n, "gushehs": g})
                else:
                    refined.append({**d, "matched_hz": None, "delta_cents": None, "ok": False})
            cand = {"ref": float(ref), "latched": latched, "score": sc, "details": refined}
            if (latched, sc) > (best["latched"], best["score"]):
                best = cand
        shift = float(cents(best["ref"], NARIMAN_REF))
        report["albums"][name] = {
            "ref": best["ref"],
            "latched": best["latched"],
            "score": best["score"],
            "n_tracks": n_tracks,
            "n_strikes": int(len(fs)),
            "degrees": best["details"],
            "shift_vs_nariman_cents": shift,
        }
        print(f"\n{name} BEST REF={best['ref']:.2f} Hz  shift={shift:+.1f}¢  latched={best['latched']}/{len(NARIMAN_CENTS)}")
        for d in best["details"]:
            flag = "✓" if d.get("ok") else "✗"
            mhz = d.get("matched_hz")
            dc = d.get("delta_cents")
            print(
                f"  {flag} {d['label']:8} expect {d['expect_hz']:7.1f}  "
                f"got {mhz if mhz else float('nan'):7.1f}  "
                f"Δ¢={dc if dc is not None else float('nan'):+7.1f}  "
                f"n={d.get('n', 0)} g={d.get('gushehs', 0)}/{n_tracks}"
            )

    out_json = OUT / "latch_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")
    return report


def plot_fingerprint(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    order = [("nariman", "C0", 2), ("alizadeh", "C1", 1), ("shekarchi", "C2", 0)]
    for name, color, y in order:
        alb = report["albums"][name]
        xs_ok, xs_bad = [], []
        for d in alb["degrees"]:
            if d.get("matched_hz") is None:
                continue
            # plot actual cents vs own REF
            c = float(cents(d["matched_hz"], alb["ref"]))
            (xs_ok if d.get("ok") else xs_bad).append(c)
        ax.scatter(xs_ok, [y] * len(xs_ok), s=90, color=color, zorder=3, label=f"{name} ✓")
        if xs_bad:
            ax.scatter(xs_bad, [y] * len(xs_bad), s=70, facecolors="none", edgecolors=color,
                       linewidths=1.6, zorder=3, label=f"{name} ✗")
    # expected template
    ax.scatter(NARIMAN_CENTS, [-0.45] * len(NARIMAN_CENTS), s=55, facecolors="none",
               edgecolors="0.2", linewidths=1.5, label="template", zorder=2)
    for c in NARIMAN_CENTS:
        ax.axvline(c, color="0.5", alpha=0.15, lw=1)
    ax.set_yticks([-0.45, 0, 1, 2])
    ax.set_yticklabels(["template", "shekarchi", "alizadeh", "nariman"])
    ax.set_xlabel("cents vs own REF")
    ax.set_title("Overnight latch fingerprint (filled=OK ≤40¢)")
    ax.set_xlim(-1100, 100)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(OUT / "latch_fingerprint.png", dpi=160)
    print(f"Wrote {OUT / 'latch_fingerprint.png'}")


def maybe_improve_cache_shekarchi():
    """If Shekarchi latch < 5, rebuild cache with wider FFT trust / lower HPS floor."""
    pass


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--only-cache", action="store_true")
    args = p.parse_args()
    if args.force_cache or args.only_cache:
        for name in ALBUMS:
            cache_album(name, force=True)
        if args.only_cache:
            raise SystemExit(0)
    report = build_report()
    plot_fingerprint(report)
    # summary latch counts
    print("\n==== LATCH SUMMARY ====")
    for name, alb in report["albums"].items():
        print(f"{name:10} REF={alb['ref']:.2f}  latched={alb['latched']}/{len(NARIMAN_CENTS)}  shift={alb['shift_vs_nariman_cents']:+.1f}¢")

#!/usr/bin/env python3
"""Rebuild consensus_template.json from cached strikes_*.npz only (no audio I/O).

Usage:
  micromamba run -n freqs python analysis/rebuild_consensus_from_cache.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"


def cents(a, b):
    return 1200.0 * np.log2(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def octave_residual(cents_arr, target):
    """Signed distance to nearest octave copy of target, in (-600, 600]."""
    c = np.asarray(cents_arr, dtype=float)
    return np.mod(c - float(target) + 600.0, 1200.0) - 600.0


def measure_cents(cents_arr, target, window=55, match=40, min_n=8, octave_fold=True):
    """Match a degree, optionally pooling ±1 octave (Sol↓ counts for Sol↑, etc.).

    matched_cents is projected into the *target* octave so Sol↑ plots near +700¢
    even when the strikes lived near Sol↓ (−500¢).

    native=True when enough strikes sit in the target octave itself (not only ±1200).
    """
    c = np.asarray(cents_arr, dtype=float)
    if not octave_fold:
        near = c[np.abs(c - target) <= window]
        if len(near) < min_n:
            return None
        ctr = float(np.median(near))
        for _ in range(3):
            cl = near[np.abs(near - ctr) <= match]
            if len(cl) < min_n:
                return None
            ctr = float(np.median(cl))
        return {
            "matched_cents": ctr,
            "delta": ctr - target,
            "n": int(len(cl)),
            "ok": abs(ctr - target) <= match,
            "octave_folded": False,
            "native": True,
            "n_native": int(len(cl)),
        }

    # native-octave strikes (no wrap)
    native_near = c[np.abs(c - target) <= window]
    n_native_raw = int(len(native_near))

    res = octave_residual(c, target)
    near_res = res[np.abs(res) <= window]
    if len(near_res) < min_n:
        return None
    ctr_res = float(np.median(near_res))
    for _ in range(3):
        cl = near_res[np.abs(near_res - ctr_res) <= match]
        if len(cl) < min_n:
            return None
        ctr_res = float(np.median(cl))
    matched = float(target + ctr_res)

    # native cluster in target octave (same match tolerance around projected center)
    if n_native_raw >= min_n:
        nctr = float(np.median(native_near))
        for _ in range(3):
            ncl = native_near[np.abs(native_near - nctr) <= match]
            if len(ncl) < min_n:
                n_native = 0
                break
            nctr = float(np.median(ncl))
        else:
            n_native = int(len(ncl))
    else:
        n_native = 0

    return {
        "matched_cents": matched,
        "delta": matched - target,
        "n": int(len(cl)),
        "ok": abs(ctr_res) <= match,
        "octave_folded": True,
        "native": n_native >= min_n,
        "n_native": n_native,
    }


def est_track_do(fs, seed, window=70, min_n=5):
    if len(fs) < min_n:
        return None
    near = fs[np.abs(cents(fs, seed)) <= window]
    if len(near) < min_n:
        return None
    ctr = float(np.median(near))
    for _ in range(3):
        cl = near[np.abs(cents(near, ctr)) <= 40]
        if len(cl) < min_n:
            return None
        ctr = float(np.median(cl))
    if abs(cents(ctr, seed)) > 60:
        return None
    return ctr


def album_cents_from_npz(path: Path, seed: float):
    d = np.load(path, allow_pickle=True)
    fs, tracks = d["f"].astype(float), d["track"].astype(str)
    parts, dos, skipped = [], {}, 0
    for tr in sorted(set(tracks)):
        do = est_track_do(fs[tracks == tr], seed)
        if do is None:
            skipped += 1
            continue
        dos[tr] = do
        parts.append(cents(fs[tracks == tr], do))
    c_all = np.concatenate(parts) if parts else np.array([])
    do_vals = np.array(list(dos.values())) if dos else np.array([seed])
    stats = {
        "seed": seed,
        "median_do": float(np.median(do_vals)),
        "std_cents": float(np.std(cents(do_vals, np.median(do_vals)))) if len(do_vals) > 1 else 0.0,
        "range_cents": float(cents(do_vals.max(), do_vals.min())) if len(do_vals) > 1 else 0.0,
        "n_tracks_used": len(dos),
        "n_tracks_skipped": skipped,
        "n_strikes": int(len(c_all)),
        "do_mode": "per_track",
    }
    return c_all, stats


def degrees_from_cents(c_all, cands, min_frac=0.003):
    dets = []
    for lab, tc in cands:
        m = measure_cents(c_all, tc, min_n=max(6, int(min_frac * max(len(c_all), 1))))
        if m is None:
            dets.append(
                {
                    "label": lab,
                    "template_cents": float(tc),
                    "ok": False,
                    "matched_cents": None,
                    "delta_cents": None,
                    "n": 0,
                    "native": False,
                    "n_native": 0,
                }
            )
        else:
            dets.append(
                {
                    "label": lab,
                    "template_cents": float(tc),
                    "ok": bool(m["ok"]),
                    "matched_cents": float(m["matched_cents"]),
                    "delta_cents": float(m["delta"]),
                    "n": int(m["n"]),
                    "native": bool(m.get("native", True)),
                    "n_native": int(m.get("n_native", m["n"])),
                }
            )
    return dets


CANDS = [
    ("Re↓", -1000),
    ("Mi↓", -900),  # caution: often +300¢ via octave fold — not low Mi
    ("Fa", -710),
    ("Sol", -490),
    ("La", -300),
    ("Si", -210),
    ("Siᵏ", -115),
    ("Do", 0),
    ("Re", 200),
    ("Mi", 400),  # includes lower octave (~−800¢) via octave fold
    ("Fa↑", 500),
    ("Sol↑", 700),
    ("La↑", 900),
]

# name -> (strikes file, how to get cents arrays / stats)
SEEDS = {
    "nariman": 266.2,
    "alizadeh": 332.2,
    "shekarchi": 441.2,
    "borumand": 327.0,
    "shahnazi": 385.2,
}


def load_lotfi_cents():
    p = OUT / "lotfi_local_do.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["cents"].astype(float), {
        "seed": float(d["seed"]) if "seed" in d.files else 326.3,
        "median_do": float(np.median(d["do_hz"])),
        "std_cents": float(np.std(cents(d["do_hz"], np.median(d["do_hz"])))),
        "range_cents": float(cents(d["do_hz"].max(), d["do_hz"].min())),
        "n_tracks_used": int(len(d["do_hz"])),
        "n_tracks_skipped": 0,
        "n_strikes": int(len(d["cents"])),
        "do_mode": "sliding_window_45s_hop_15s",
    }


def load_ebadi_cents():
    p = OUT / "ebadi_local_do.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["cents"].astype(float), {
        "seed": float(d["seed"]) if "seed" in d.files else float(np.median(d["do_hz"])),
        "median_do": float(np.median(d["do_hz"])),
        "std_cents": float(np.std(cents(d["do_hz"], np.median(d["do_hz"])))),
        "range_cents": float(cents(d["do_hz"].max(), d["do_hz"].min())),
        "n_tracks_used": int(len(d["do_hz"])),
        "n_tracks_skipped": 0,
        "n_strikes": int(len(d["cents"])),
        "do_mode": "sliding_window_45s_hop_15s_mahur_only",
    }


def load_kasaei_radif_cents():
    filt = json.loads((OUT / "kasaei_radif_filter.json").read_text())
    d = np.load(OUT / "strikes_kasaei_radif.npz", allow_pickle=True)
    fs, tr = d["f"].astype(float), d["track"].astype(str)
    do_by = {k["track"]: k["do"] for k in filt["kept"]}
    parts = [cents(fs[tr == name], do) for name, do in do_by.items() if name in set(tr)]
    c_all = np.concatenate(parts) if parts else np.array([])
    dos = [k["do"] for k in filt["kept"]]
    return c_all, {
        "seed": filt["seed_do"],
        "median_do": filt["median_do"],
        "std_cents": float(np.std([cents(v, filt["median_do"]) for v in dos])) if len(dos) > 1 else 0.0,
        "range_cents": float(cents(max(dos), min(dos))) if len(dos) > 1 else 0.0,
        "n_tracks_used": len(dos),
        "n_tracks_skipped": len(filt.get("skipped", [])) + int(filt.get("skipped_shur", 0)),
        "n_strikes": int(len(c_all)),
        "do_mode": "per_track_setar_mahoor_radif",
    }


def load_karimi_cents():
    filt = json.loads((OUT / "karimi_filter.json").read_text())
    d = np.load(OUT / "strikes_karimi.npz", allow_pickle=True)
    fs, tr = d["f"].astype(float), d["track"].astype(str)
    do_by = {k["track"]: k["do"] for k in filt["kept"]}
    parts = [cents(fs[tr == name], do) for name, do in do_by.items() if name in set(tr)]
    c_all = np.concatenate(parts) if parts else np.array([])
    dos = [k["do"] for k in filt["kept"]]
    return c_all, {
        "seed": filt["seed_do"],
        "median_do": filt["median_do"],
        "std_cents": float(np.std([cents(v, filt["median_do"]) for v in dos])) if len(dos) > 1 else 0.0,
        "range_cents": float(cents(max(dos), min(dos))) if len(dos) > 1 else 0.0,
        "n_tracks_used": len(dos),
        "n_tracks_skipped": len(filt.get("skipped", [])),
        "n_strikes": int(len(c_all)),
        "do_mode": "per_track_vocal_mahur",
    }


def load_saba_cents():
    p = OUT / "saba_local_do.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["cents"].astype(float), {
        "seed": float(d["seed"]) if "seed" in d.files else float(np.median(d["do_hz"])),
        "median_do": float(np.median(d["do_hz"])),
        "std_cents": float(np.std(cents(d["do_hz"], np.median(d["do_hz"])))),
        "range_cents": float(cents(d["do_hz"].max(), d["do_hz"].min())),
        "n_tracks_used": int(len(d["do_hz"])),
        "n_tracks_skipped": 0,
        "n_strikes": int(len(d["cents"])),
        "do_mode": "sliding_window_45s_hop_15s_setar_mahoor",
    }


def load_lotfi_mahur_cents():
    filt = json.loads((OUT / "lotfi_playlist_filter.json").read_text())
    pl = np.load(OUT / "strikes_lotfi_playlist.npz", allow_pickle=True)
    fs, tr = pl["f"].astype(float), pl["track"].astype(str)
    do_by = {k["track"]: k["do"] for k in filt["kept"]}
    parts = [cents(fs[tr == name], do) for name, do in do_by.items()]
    c_all = np.concatenate(parts) if parts else np.array([])
    dos = list(do_by.values())
    return c_all, {
        "seed": filt["seed_do"],
        "median_do": filt["median_do"],
        "std_cents": float(np.std([cents(v, filt["median_do"]) for v in dos])) if len(dos) > 1 else 0.0,
        "range_cents": float(cents(max(dos), min(dos))) if len(dos) > 1 else 0.0,
        "n_tracks_used": len(dos),
        "n_tracks_skipped": len(filt.get("skipped", [])) + len(filt.get("title_skipped_not_downloaded", [])),
        "n_strikes": int(len(c_all)),
        "do_mode": "per_track_curated_playlist",
    }


def load_neydavud_cents():
    filt = json.loads((OUT / "neydavud_filter.json").read_text())
    d = np.load(OUT / "strikes_neydavud.npz", allow_pickle=True)
    fs, tr = d["f"].astype(float), d["track"].astype(str)
    do_by = {k["track"]: k["do"] for k in filt["kept"]}
    parts = [cents(fs[tr == name], do) for name, do in do_by.items() if name in set(tr)]
    c_all = np.concatenate(parts) if parts else np.array([])
    dos = [k["do"] for k in filt["kept"]]
    return c_all, {
        "seed": filt["seed_do"],
        "median_do": filt["median_do"],
        "std_cents": float(np.std([cents(v, filt["median_do"]) for v in dos])) if len(dos) > 1 else 0.0,
        "range_cents": float(cents(max(dos), min(dos))) if len(dos) > 1 else 0.0,
        "n_tracks_used": len(dos),
        "n_tracks_skipped": len(filt.get("skipped", [])) + len(filt.get("missing_download", [])),
        "n_strikes": int(len(c_all)),
        "do_mode": "per_track_quality_filtered",
    }


def load_forutan_cents():
    # prefer seed from prior consensus / filter
    prev = OUT / "consensus_template.json"
    seed = 225.0
    if prev.exists():
        old = json.loads(prev.read_text())
        seed = old.get("forutan_filter", {}).get("seed_do") or old["do_stats"].get("forutan", {}).get("seed", seed)
    return album_cents_from_npz(OUT / "strikes_forutan.npz", float(seed))


def rebuild(write: bool = True):
    artist_cents = {}
    do_stats = {}

    for name, seed in SEEDS.items():
        c_all, stats = album_cents_from_npz(OUT / f"strikes_{name}.npz", seed)
        artist_cents[name] = c_all
        do_stats[name] = stats

    for name, loader in [
        ("lotfi", load_lotfi_cents),
        ("lotfi_mahur", load_lotfi_mahur_cents),
        ("forutan", load_forutan_cents),
        ("neydavud", load_neydavud_cents),
        ("ebadi", load_ebadi_cents),
        ("kasaei_radif", load_kasaei_radif_cents),
        ("saba", load_saba_cents),
        ("karimi", load_karimi_cents),
    ]:
        c_all, stats = loader()
        # forutan loader returns per_track mode already
        if name == "forutan":
            stats["do_mode"] = "per_track_mahur_only"
        artist_cents[name] = c_all
        do_stats[name] = stats

    artist_details = {n: degrees_from_cents(c, CANDS) for n, c in artist_cents.items()}

    by_slot = defaultdict(dict)
    for name, dets in artist_details.items():
        for d in dets:
            if d["ok"] and d["matched_cents"] is not None:
                by_slot[d["label"]][name] = d["matched_cents"]

    n_art = len(artist_details)
    consensus, solo = [], []
    for lab, tc in CANDS:
        vals = by_slot.get(lab, {})
        if lab == "Do":
            consensus.append(
                {
                    "label": "Do",
                    "name": "Do",
                    "cents": 0.0,
                    "n_artists": n_art,
                    "per_artist": {a: 0.0 for a in artist_details},
                }
            )
            continue
        if len(vals) < 2:
            if vals:
                solo.append(
                    {
                        "label": lab,
                        "name": lab,
                        "cents": round(list(vals.values())[0], 1),
                        "artist": list(vals.keys())[0],
                    }
                )
            continue
        med = float(np.median(list(vals.values())))
        consensus.append(
            {
                "label": lab,
                "name": lab,
                "cents": round(med, 1),
                "n_artists": len(vals),
                "per_artist": {k: round(v, 1) for k, v in vals.items()},
            }
        )

    # preserve filter metadata if present
    prev = {}
    prev_path = OUT / "consensus_template.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())

    out = {
        "method": "rebuild_consensus_from_cache.py — local-Do cents, octave-folded degree match",
        "min_artists": 2,
        "octave_fold": True,
        "do_stats": do_stats,
        "slots": consensus,
        "solo_excluded": solo,
        "albums": {
            name: {
                "median_do": do_stats[name]["median_do"],
                "do_drift_std_cents": do_stats[name]["std_cents"],
                "do_drift_range_cents": do_stats[name]["range_cents"],
                "n_tracks_used": do_stats[name]["n_tracks_used"],
                "n_tracks_skipped": do_stats[name]["n_tracks_skipped"],
                "details": artist_details[name],
                "n_strikes": do_stats[name]["n_strikes"],
                "do_mode": do_stats[name]["do_mode"],
            }
            for name in artist_details
        },
        "lotfi_mahur_filter": prev.get("lotfi_mahur_filter"),
        "forutan_filter": prev.get("forutan_filter"),
        "neydavud_filter": prev.get("neydavud_filter"),
        "ebadi_filter": prev.get("ebadi_filter"),
        "shahnazi_filter": prev.get("shahnazi_filter")
        or (
            json.loads((OUT / "shahnazi_filter.json").read_text())
            if (OUT / "shahnazi_filter.json").exists()
            else None
        ),
        "kasaei_radif_filter": (
            json.loads((OUT / "kasaei_radif_filter.json").read_text())
            if (OUT / "kasaei_radif_filter.json").exists()
            else prev.get("kasaei_radif_filter")
        ),
        "saba_filter": prev.get("saba_filter")
        or (
            json.loads((OUT / "saba_filter.json").read_text())
            if (OUT / "saba_filter.json").exists()
            else None
        ),
        "karimi_filter": (
            json.loads((OUT / "karimi_filter.json").read_text())
            if (OUT / "karimi_filter.json").exists()
            else prev.get("karimi_filter")
        ),
    }
    if write:
        prev_path.write_text(json.dumps(out, indent=2))
    return out


def cache_status():
    needed = [
        "strikes_nariman.npz",
        "strikes_alizadeh.npz",
        "strikes_shekarchi.npz",
        "strikes_borumand.npz",
        "strikes_forutan.npz",
        "strikes_neydavud.npz",
        "strikes_shahnazi.npz",
        "strikes_lotfi_playlist.npz",
        "lotfi_local_do.npz",
        "lotfi_playlist_filter.json",
        "neydavud_filter.json",
        "ebadi_local_do.npz",
        "shahnazi_filter.json",
        "strikes_kasaei_radif.npz",
        "kasaei_radif_filter.json",
        "saba_local_do.npz",
        "saba_filter.json",
        "strikes_karimi.npz",
        "karimi_filter.json",
        "consensus_template.json",
    ]
    rows = []
    for name in needed:
        p = OUT / name
        rows.append((name, p.exists(), p.stat().st_size if p.exists() else 0))
    return rows


if __name__ == "__main__":
    print("cache status:")
    for name, ok, sz in cache_status():
        print(f"  {'✓' if ok else '✗'} {name:35} {sz/1e3:.1f} KB" if ok else f"  ✗ {name}")
    out = rebuild(write=True)
    print(f"\nrebuilt consensus: {len(out['slots'])} slots, {len(out['albums'])} artists")
    print("→", OUT / "consensus_template.json")

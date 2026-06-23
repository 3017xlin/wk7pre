#!/usr/bin/env python3
"""
EDA on physical PT files: building count, volume/surface point counts.
DBSCAN on stl_centers XY to count buildings.
BIC-based model selection for regression.
KDE for smooth distributions.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
import torch
from tqdm import tqdm


# ── DBSCAN params ─────────────────────────────────────────────
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 10

# ── worker globals ────────────────────────────────────────────
def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
        os.environ[k] = "1"
    torch.set_num_threads(1)


def _process_one(pt_path_str: str) -> dict | None:
    pt_path = Path(pt_path_str)
    try:
        d = torch.load(str(pt_path), map_location="cpu", weights_only=False)

        case_name = d["case_name"]
        n_volume = int(d["volume_fields"].shape[0])
        n_surface = int(d["surface_fields"].shape[0])

        # DBSCAN on stl_centers XY projection
        stl_centers = d["stl_centers"].numpy()  # (F, 3)
        xy = stl_centers[:, :2].astype(np.float64)
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
        # count clusters, ignore noise (-1)
        unique = set(labels)
        unique.discard(-1)
        n_buildings = len(unique)

        return {
            "case_name": case_name,
            "n_buildings": n_buildings,
            "n_volume": n_volume,
            "n_surface": n_surface,
        }
    except Exception as e:
        print(f"[WARN] {pt_path.name}: {e}")
        return None


# ── regression models ─────────────────────────────────────────
def _linear(x, a, b):
    return a * x + b

def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def _power(x, a, b):
    return a * np.power(x, b)

def _log(x, a, b):
    return a * np.log(x) + b

def _sqrt(x, a, b):
    return a * np.sqrt(x) + b


MODELS = [
    ("linear",    _linear,    2, [1.0, 0.0]),
    ("quadratic", _quadratic, 3, [1.0, 1.0, 0.0]),
    ("power",     _power,     2, [1.0, 1.0]),
    ("log",       _log,       2, [1.0, 0.0]),
    ("sqrt",      _sqrt,      2, [1.0, 0.0]),
]


def _fit_and_score(x, y):
    """Fit all candidate models, return sorted by BIC."""
    n = len(x)
    results = []
    for name, func, k, p0 in MODELS:
        try:
            popt, _ = optimize.curve_fit(func, x, y, p0=p0, maxfev=10000)
            y_pred = func(x, *popt)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # BIC = n*ln(ss_res/n) + k*ln(n)
            mse = ss_res / n
            if mse > 0:
                bic = n * np.log(mse) + k * np.log(n)
            else:
                bic = -np.inf
            results.append({
                "name": name, "func": func, "popt": popt,
                "k": k, "r2": r2, "bic": bic,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r["bic"])
    return results


# ── plotting helpers ──────────────────────────────────────────
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]


def plot_building_histogram(data, out_dir):
    counts = np.array([r["n_buildings"] for r in data])
    fig, ax = plt.subplots(figsize=(12, 6))

    lo, hi = int(counts.min()), int(counts.max())
    bins = np.arange(lo, hi + 2) - 0.5
    ax.hist(counts, bins=bins, color="#2196F3", edgecolor="white", linewidth=0.5)

    mean_val = counts.mean()
    med_val = np.median(counts)
    std_val = counts.std()
    ax.axvline(mean_val, color="red", ls="--", lw=1.5, label=f"mean = {mean_val:.1f}")
    ax.axvline(med_val, color="orange", ls="-.", lw=1.5, label=f"median = {med_val:.0f}")
    ax.set_xlabel("Number of buildings", fontsize=12)
    ax.set_ylabel("Number of cases", fontsize=12)
    ax.set_title(f"Building count distribution (n={len(counts)}, std={std_val:.1f}, "
                 f"min={lo}, max={hi})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(range(lo, hi + 1, max(1, (hi - lo) // 20)))
    fig.tight_layout()
    fig.savefig(out_dir / "building_count_histogram.png", dpi=200)
    plt.close(fig)


def plot_kde_distribution(values, label, unit, out_dir, filename):
    values = np.array(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 6))

    # histogram (faint background)
    ax.hist(values, bins=80, density=True, color="#BBDEFB", edgecolor="white",
            linewidth=0.3, alpha=0.6, label="histogram")

    # KDE smooth curve
    kde = gaussian_kde(values, bw_method="scott")
    x_grid = np.linspace(values.min(), values.max(), 500)
    ax.plot(x_grid, kde(x_grid), color="#1565C0", lw=2, label="KDE")

    mean_val = values.mean()
    med_val = np.median(values)
    std_val = values.std()
    ax.axvline(mean_val, color="red", ls="--", lw=1.5,
               label=f"mean = {mean_val:,.0f}")
    ax.axvline(med_val, color="orange", ls="-.", lw=1.5,
               label=f"median = {med_val:,.0f}")
    ax.axvline(mean_val - std_val, color="gray", ls=":", lw=1,
               label=f"±1σ = {std_val:,.0f}")
    ax.axvline(mean_val + std_val, color="gray", ls=":", lw=1)

    ax.set_xlabel(f"{label} ({unit})", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(f"{label} distribution (n={len(values)}, min={values.min():,.0f}, "
                 f"max={values.max():,.0f})", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def plot_regression(x, y, xlabel, ylabel, out_dir, filename):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # filter x > 0 for log/power models
    mask = x > 0
    x, y = x[mask], y[mask]

    fits = _fit_and_score(x, y)
    if not fits:
        print(f"[WARN] no fits converged for {filename}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x, y, s=4, alpha=0.15, color="#90A4AE", rasterized=True, label="data")

    x_smooth = np.linspace(x.min(), x.max(), 300)
    best = fits[0]

    for i, fit in enumerate(fits):
        is_best = (i == 0)
        try:
            y_smooth = fit["func"](x_smooth, *fit["popt"])
            style = dict(lw=2.5, alpha=1.0) if is_best else dict(lw=1.2, alpha=0.5, ls="--")
            tag = " [BEST]" if is_best else ""
            ax.plot(x_smooth, y_smooth, color=COLORS[i % len(COLORS)], **style,
                    label=f'{fit["name"]}{tag}  R²={fit["r2"]:.4f}  BIC={fit["bic"]:.0f}')
        except Exception:
            continue

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs {xlabel} — best model: {best['name']} "
                 f"(R²={best['r2']:.4f})", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def compute_percentile_stats(values, name):
    values = np.array(values, dtype=np.float64)
    pcts = list(range(5, 101, 5))
    pct_values = np.percentile(values, pcts)
    return {
        "name": name,
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "percentiles": {str(p): float(v) for p, v in zip(pcts, pct_values)},
    }


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with physical .pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for plots and JSON")
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {input_dir}")
    print(f"[eda] {len(pt_files)} PT files, {workers} workers")

    t0 = time.time()

    if workers <= 1:
        _init_worker()
        results = [_process_one(str(p)) for p in tqdm(pt_files, desc="scan")]
    else:
        pool = mp.Pool(workers, initializer=_init_worker)
        results = list(tqdm(
            pool.imap_unordered(_process_one, [str(p) for p in pt_files], chunksize=4),
            total=len(pt_files), desc="scan",
        ))
        pool.close()
        pool.join()

    data = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"[eda] scan done in {elapsed:.0f}s — {len(data)} / {len(pt_files)} succeeded")

    if not data:
        print("[eda] no data, exiting")
        return

    # ── extract arrays ────────────────────────────────────────
    n_buildings = [r["n_buildings"] for r in data]
    n_volume = [r["n_volume"] for r in data]
    n_surface = [r["n_surface"] for r in data]

    # ── JSON: case-level mapping ──────────────────────────────
    case_map = {}
    for r in data:
        case_map[r["case_name"]] = {
            "n_buildings": r["n_buildings"],
            "n_volume": r["n_volume"],
            "n_surface": r["n_surface"],
        }
    with open(output_dir / "case_building_counts.json", "w") as f:
        json.dump(case_map, f, indent=2)
    print(f"[eda] saved case_building_counts.json ({len(case_map)} cases)")

    # ── JSON: distribution stats ──────────────────────────────
    stats = {
        "n_buildings": compute_percentile_stats(n_buildings, "n_buildings"),
        "n_volume": compute_percentile_stats(n_volume, "n_volume"),
        "n_surface": compute_percentile_stats(n_surface, "n_surface"),
    }
    with open(output_dir / "distribution_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[eda] saved distribution_stats.json")

    # ── JSON: regression results ──────────────────────────────
    x_b = np.array(n_buildings, dtype=np.float64)
    mask_pos = x_b > 0
    reg_results = {}
    for y_arr, y_name in [(n_volume, "n_volume"), (n_surface, "n_surface")]:
        y_np = np.array(y_arr, dtype=np.float64)
        fits = _fit_and_score(x_b[mask_pos], y_np[mask_pos])
        reg_results[f"buildings_vs_{y_name}"] = [
            {"model": f["name"], "r2": round(f["r2"], 6),
             "bic": round(f["bic"], 2),
             "params": [round(float(p), 6) for p in f["popt"]]}
            for f in fits
        ]
    with open(output_dir / "regression_results.json", "w") as f:
        json.dump(reg_results, f, indent=2)
    print(f"[eda] saved regression_results.json")

    # ── plots ─────────────────────────────────────────────────
    print("[eda] generating plots...")

    plot_building_histogram(data, output_dir)
    print("  -> building_count_histogram.png")

    plot_kde_distribution(n_volume, "Volume cell count", "cells",
                          output_dir, "volume_count_distribution.png")
    print("  -> volume_count_distribution.png")

    plot_kde_distribution(n_surface, "Surface cell count", "cells",
                          output_dir, "surface_count_distribution.png")
    print("  -> surface_count_distribution.png")

    plot_regression(n_buildings, n_volume,
                    "Number of buildings", "Volume cell count",
                    output_dir, "buildings_vs_volume.png")
    print("  -> buildings_vs_volume.png")

    plot_regression(n_buildings, n_surface,
                    "Number of buildings", "Surface cell count",
                    output_dir, "buildings_vs_surface.png")
    print("  -> buildings_vs_surface.png")

    total = time.time() - t0
    print(f"[eda] all done in {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

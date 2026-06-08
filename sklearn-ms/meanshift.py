"""Reference mean-shift clustering of Chicago crime coordinates using scikit-learn.

Clusters on the planar X / Y COORDINATE columns (units: feet), which is the right
metric space for Euclidean mean-shift. Intended as a CPU baseline to validate /
benchmark against the Ray-Shift GPU implementation.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="../data/chicago_crime.data",
                   help="Path to the input .data file (default: ../data/chicago_crime.data)")
    p.add_argument("--output", default="clusters.csv",
                   help="Where to write CASE#,X,Y,cluster_label (default: clusters.csv)")
    p.add_argument("--sample", default="10000",
                   help="Number of points to cluster, or 'all'/'0' for the full set (default: 10000)")
    p.add_argument("--bandwidth", type=float, default=None,
                   help="Mean-shift bandwidth in feet. If omitted, it is estimated.")
    p.add_argument("--quantile", type=float, default=0.2,
                   help="Quantile for estimate_bandwidth when --bandwidth is omitted (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible sampling (default: 42)")
    return p.parse_args()


def load_xy(path):
    """Load CASE#/X/Y, dropping rows with missing or non-numeric coordinates."""
    df = pd.read_csv(path, usecols=["CASE#", "X COORDINATE", "Y COORDINATE"])
    df = df.rename(columns={"X COORDINATE": "X", "Y COORDINATE": "Y"})
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)
    print(f"Loaded {before} rows, {len(df)} valid after dropping missing coordinates.")
    return df


def take_sample(df, sample, seed):
    if str(sample).lower() in ("all", "0"):
        return df
    n = int(sample)
    if n >= len(df):
        return df
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sampled)} of {len(df)} points (seed={seed}).")
    return sampled


def main():
    args = parse_args()

    df = load_xy(args.input)
    df = take_sample(df, args.sample, args.seed)
    X = df[["X", "Y"]].to_numpy(dtype=np.float64)
    print(f"Clustering {len(X)} points on X/Y coordinates (feet).")

    bw = args.bandwidth
    if bw is None:
        n_samples = min(500, len(X))
        bw = estimate_bandwidth(X, quantile=args.quantile, n_samples=n_samples,
                                random_state=args.seed)
        print(f"Estimated bandwidth: {bw:.2f} feet "
              f"(quantile={args.quantile}, n_samples={n_samples}).")
    else:
        print(f"Using bandwidth: {bw:.2f} feet.")

    if not bw or bw <= 0:
        raise SystemExit("Bandwidth is non-positive; pass --bandwidth explicitly.")

    ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
    t0 = time.perf_counter()
    labels = ms.fit_predict(X)
    elapsed = time.perf_counter() - t0

    centers = ms.cluster_centers_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nMean-shift fit completed in {elapsed:.2f} s.")
    print(f"Found {len(unique)} clusters.\n")
    print(f"{'cluster':>7}  {'count':>8}  {'center_x':>14}  {'center_y':>14}")
    order = np.argsort(-counts)
    for i in order:
        c = unique[i]
        print(f"{c:>7}  {counts[i]:>8}  {centers[c, 0]:>14.1f}  {centers[c, 1]:>14.1f}")

    out = df.copy()
    out["cluster_label"] = labels
    out[["CASE#", "X", "Y", "cluster_label"]].to_csv(args.output, index=False)
    print(f"\nWrote per-point labels to {args.output}")


if __name__ == "__main__":
    main()

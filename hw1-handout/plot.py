import argparse
from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

# This script intentionally avoids external dependencies like pandas.
# Input CSV (produced by `parse.py`):
# columns: n, p, avg_query_distance, max_level, avg_height


def read_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "n": int(float(row["n"])),
                "p": float(row["p"]),
                "avg_query_distance": float(row["avg_query_distance"]),
                "max_level": float(row["max_level"]),
                "avg_height": float(row["avg_height"]),
            })
    return rows


def plot_one(p_list, y_list, xlabel, ylabel, title, out_path: Path):
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(p_list, y_list, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="result.csv", help="input CSV (default: result.csv)")
    ap.add_argument("--outdir", dest="outdir", default="figs", help="output directory (default: figs)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(inp)

    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)

    for n in sorted(by_n.keys()):
        data = sorted(by_n[n], key=lambda r: r["p"])
        p = [r["p"] for r in data]

        plot_one(
            p,
            [r["avg_query_distance"] for r in data],
            "p",
            "average search length (visited nodes)",
            f"n={n}: avg_query_distance vs p",
            outdir / f"n{n}_avg_query_distance.png",
        )

        plot_one(
            p,
            [r["max_level"] for r in data],
            "p",
            "max level (number of layers)",
            f"n={n}: max_level vs p",
            outdir / f"n{n}_max_level.png",
        )

        plot_one(
            p,
            [r["avg_height"] for r in data],
            "p",
            "average tower height",
            f"n={n}: avg_height vs p",
            outdir / f"n{n}_avg_height.png",
        )

    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()

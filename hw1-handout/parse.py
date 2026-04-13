import re
import csv
import argparse
from pathlib import Path

# Parse output produced by `test-main` (see `test-main.cc:73` and `test-main.cc:76`).
# Expected two lines per (n, p) group:
#   (element#=50, p=0.500000) average query distance = 12.345600
#   (element#=50, p=0.500000) max_level = 7, average_height = 1.937500

LINE1 = re.compile(r"^\(element#=(?P<n>\d+), p=(?P<p>[0-9eE+\-\.]+)\) average query distance = (?P<avgqd>[0-9eE+\-\.]+)\s*$")
LINE2 = re.compile(r"^\(element#=(?P<n>\d+), p=(?P<p>[0-9eE+\-\.]+)\) max_level = (?P<maxlvl>\d+), average_height = (?P<avgh>[0-9eE+\-\.]+)\s*$")


def parse_raw(text: str):
    rows = []
    pending = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m1 = LINE1.match(line)
        if m1:
            key = (int(m1.group("n")), float(m1.group("p")))
            pending[key] = {
                "n": key[0],
                "p": key[1],
                "avg_query_distance": float(m1.group("avgqd")),
            }
            continue

        m2 = LINE2.match(line)
        if m2:
            key = (int(m2.group("n")), float(m2.group("p")))
            base = pending.pop(key, {"n": key[0], "p": key[1]})
            base["max_level"] = int(m2.group("maxlvl"))
            base["avg_height"] = float(m2.group("avgh"))
            rows.append(base)
            continue

    # ignore incomplete groups silently
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="raw.txt", help="input raw text file (default: raw.txt)")
    ap.add_argument("--out", dest="out", default="result.csv", help="output csv file (default: result.csv)")
    args = ap.parse_args()

    inp_path = Path(args.inp)
    out_path = Path(args.out)

    text = inp_path.read_text(encoding="utf-8", errors="ignore")
    rows = parse_raw(text)

    rows.sort(key=lambda r: (r["n"], r["p"]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "p", "avg_query_distance", "max_level", "avg_height"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import glob, os, pandas as pd, argparse, gzip, pathlib, tempfile

def load_one(path: str) -> pd.DataFrame:
    """Load a rocprof counter CSV (plain or .gz) and add `source` col."""
    if path.endswith(".gz"):
        df = pd.read_csv(gzip.open(path, "rt"))
    else:
        df = pd.read_csv(path)
    df["source"] = os.path.basename(path)
    return df

def combine_csvs(dir_: str, pattern: str = "*_counters.csv*") -> pd.DataFrame:
    """Glob `dir_`/`pattern`, read each CSV, concat, return DataFrame."""
    files = sorted(glob.glob(os.path.join(dir_, pattern)))
    if not files:
        raise FileNotFoundError(f"No files match {pattern!r} in {dir_!r}")
    print(f"▶  Found {len(files)} counter CSVs")
    return pd.concat([load_one(f) for f in files], ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing rocprof counter CSVs")
    ap.add_argument("-o", "--out", default="all_counters.csv",
                    help="merged CSV name (default: all_counters.csv)")
    args = ap.parse_args()

    df = combine_csvs(args.run_dir)

    # save to a temp file first so partial writes never clobber previous runs
    tmp = pathlib.Path(tempfile.gettempdir()) / (args.out + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(args.out)           # atomic rename

    print(f"✅  Merged counters → {args.out}  ({len(df):,} lines)")

if __name__ == "__main__":
    main()


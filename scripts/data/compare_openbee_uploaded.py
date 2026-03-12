#!/usr/bin/env python3
"""Compare local vs HF file counts for openbee_images subdirs."""
import sys
from pathlib import Path

IMAGE_ROOT = Path("/ov2/dataset_mid_source/Open-Bee/Honey-Data-15M_images")
NAMES = ["AI2D_InternVL", "CMM-Math", "CoSyn_Chemical", "CoSyn_Circuit", "InterGPS"]

def count_local(name: str) -> int:
    d = IMAGE_ROOT / name
    if not d.is_dir():
        return -1
    return sum(1 for _ in d.rglob("*") if _.is_file())

def main():
    from huggingface_hub import list_repo_files
    repo_id = "OV2-VideoQA/captions"
    print("Counting local files...")
    local_counts = {n: count_local(n) for n in NAMES}
    print("Fetching HF repo file list (once)...")
    all_files = list(list_repo_files(repo_id, repo_type="dataset"))
    prefix = "openbee_images/"
    hf_counts = {}
    for name in NAMES:
        p = prefix + name + "/"
        hf_counts[name] = sum(1 for f in all_files if f.startswith(p))
    print("\nDirectory          Local    HF    Status")
    print("-" * 45)
    all_ok = True
    for name in NAMES:
        local_n = local_counts[name]
        hf_n = hf_counts[name]
        if local_n == hf_n:
            status = "OK"
        else:
            status = "INCOMPLETE" if hf_n < local_n else "EXTRA?"
            all_ok = False
        print(f"  {name:18}  {local_n:5}  {hf_n:5}  {status}")
    print()
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Output openbee_images folder names that are COMPLETE on HF (local count == HF count).
Used by upload script RESUME=1: only skip these names, re-upload incomplete or missing."""
import sys
from pathlib import Path

IMAGE_ROOT = Path("/ov2/dataset_mid_source/Open-Bee/Honey-Data-15M_images")
NAMES = [
    "AI2D_InternVL", "CMM-Math", "CoSyn_Chemical", "CoSyn_Circuit", "CoSyn_Graphic",
    "CoSyn_Math", "CoSyn_nutrition", "Docmatix", "Geometry3K", "GeomVerse", "GeoQA+",
    "InterGPS", "MathV360K_TQA", "MAVIS_Function", "MAVIS-Geo", "MAVIS-Metagen",
    "MMChem", "PathVQA", "PMC-VQA", "ScienceQA", "TabMWP", "TQA", "UniGeo", "VQA-RAD",
]

def count_local(name: str) -> int:
    d = IMAGE_ROOT / name
    if not d.is_dir():
        return -1
    return sum(1 for _ in d.rglob("*") if _.is_file())

def main():
    from huggingface_hub import list_repo_files
    repo_id = "OV2-VideoQA/captions"
    # Optional: only check names that exist on HF (list_repo_tree first) to save time
    print("Counting local files...", file=sys.stderr)
    local_counts = {n: count_local(n) for n in NAMES}
    print("Fetching HF repo file list...", file=sys.stderr)
    all_files = list(list_repo_files(repo_id, repo_type="dataset"))
    prefix = "openbee_images/"
    complete = []
    for name in NAMES:
        p = prefix + name + "/"
        hf_n = sum(1 for f in all_files if f.startswith(p))
        local_n = local_counts[name]
        # >= 因为超过 1 万文件的目录会拆成 name/0/, name/1/ 上传，HF 总数可能略多
        if local_n >= 0 and hf_n >= local_n:
            complete.append(name)
    for name in complete:
        print(name)
    return 0

if __name__ == "__main__":
    sys.exit(main())

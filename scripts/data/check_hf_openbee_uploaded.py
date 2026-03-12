#!/usr/bin/env python3
from huggingface_hub import list_repo_tree
from huggingface_hub.hf_api import RepoFolder

REPO_ID = "OV2-VideoQA/captions"
PATH = "openbee_images"

def main():
    tree = list(list_repo_tree(REPO_ID, path_in_repo=PATH, recursive=False, repo_type="dataset"))
    dirs = [e.path.split("/")[-1] for e in tree if isinstance(e, RepoFolder)]
    dirs.sort()
    print("uploaded_count:", len(dirs))
    for d in dirs:
        print(d)
    return dirs

if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
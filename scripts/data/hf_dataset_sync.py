#!/usr/bin/env python3
"""
Hugging Face 数据集上传/下载（同一脚本，子命令 upload / download）。
默认仓库：OV2-VideoQA/captions。

使用前请先登录：
    pip install "huggingface_hub>=0.34.0,<1.0"
    huggingface-cli login
    Token: https://huggingface.co/settings/tokens

上传示例：
    # 上传整个目录
    python scripts/data/hf_dataset_sync.py upload --repo-id OV2-VideoQA/captions --dataset-dir /path/to/dir

    # 上传单个/多个文件
    python scripts/data/hf_dataset_sync.py upload --file /path/to/file.jsonl
    python scripts/data/hf_dataset_sync.py upload --file /path/to/a.jsonl --file /path/to/b.jsonl --path-in-repo captions

下载示例：
    # 下载整个仓库
    python scripts/data/hf_dataset_sync.py download --local-dir ./data/captions

    # 下载指定文件（可多个）
    python scripts/data/hf_dataset_sync.py download --file 60s_part01_N400000_azure_20260224_233403.jsonl --file 30s_part01_N500000_azure_20260224_233403.jsonl --local-dir ./output
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token, hf_hub_download, snapshot_download


DEFAULT_REPO_ID = "OV2-VideoQA/captions"
# 脚本内固定代理，不使用系统环境变量中的代理
DEFAULT_PROXY = "http://172.16.5.79:18000"


def _setup_proxy(cli_proxy: str | None = None):
    # 优先使用命令行 --proxy，否则使用脚本内默认代理，不读系统 proxy
    proxy = cli_proxy if cli_proxy is not None else DEFAULT_PROXY
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    print(f"使用代理: {proxy}")


def _ensure_hf_login(strict_check: bool = False) -> HfApi:
    token = get_token()
    if not token:
        raise SystemExit(
            "未检测到 Hugging Face 登录信息。"
            "请先执行 huggingface-cli login，或设置 HF_TOKEN。"
        )

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname") or "unknown"
        print(f"HF 登录检查通过: {username}")
    except Exception as e:
        if strict_check:
            raise SystemExit(
                "HF 登录检查失败。可能是 token 失效，或网络/代理不可用。"
                f"原始错误: {e}"
            ) from e
        print(
            "警告: HF 在线登录校验失败（可能是网络/代理问题），"
            "已检测到本地 token，继续执行。"
            f"原始错误: {e}"
        )
    return api


def cmd_upload(args):
    _setup_proxy(args.proxy)
    api = _ensure_hf_login(args.strict_login_check)
    repo_id = args.repo_id or DEFAULT_REPO_ID

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"仓库已就绪: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"创建/检查仓库时: {e}")

    if args.file:
        files = [Path(p) for p in args.file]
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"路径不是文件: {file_path}")

        prefix = (args.path_in_repo or "").strip("/")
        if len(files) == 1:
            file_path = files[0]
            path_in_repo = args.path_in_repo if args.path_in_repo else file_path.name
            print(f"正在上传文件: {file_path} -> {repo_id}/{path_in_repo} ...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("文件上传完成.")
        else:
            print(f"准备上传 {len(files)} 个文件到仓库 {repo_id} ...")
            if prefix:
                print(f"目标目录前缀: {prefix}/")
            for file_path in files:
                target_path = f"{prefix}/{file_path.name}" if prefix else file_path.name
                print(f"正在上传: {file_path} -> {repo_id}/{target_path} ...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=target_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            print("多文件上传完成.")
    else:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"目录不存在: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise ValueError(f"路径不是目录: {dataset_dir}")
        print(f"正在上传目录: {dataset_dir} -> {repo_id} ...")
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("目录上传完成.")

    print(f"数据集地址: https://huggingface.co/datasets/{repo_id}")


def cmd_download(args):
    _setup_proxy(args.proxy)
    _ensure_hf_login(args.strict_login_check)
    repo_id = args.repo_id or DEFAULT_REPO_ID
    local_dir = Path(args.local_dir or ".").resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [f.strip("/") for f in args.file]
        print(f"准备从 {repo_id} 下载 {len(files)} 个文件到 {local_dir} ...")
        for path_in_repo in files:
            print(f"正在下载: {path_in_repo} ...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                local_dir=str(local_dir),
                force_download=args.force_download,
            )
            print(f"  -> {local_path}")
        print("多文件下载完成.")
    else:
        print(f"正在下载整个仓库 {repo_id} 到 {local_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            force_download=args.force_download,
        )
        print("仓库下载完成.")

    print(f"本地目录: {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face 数据集上传/下载（默认仓库: OV2-VideoQA/captions）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="upload 或 download")

    # ---------- upload ----------
    up = subparsers.add_parser("upload", help="上传文件或目录")
    up.add_argument("--dataset-dir", type=str, default=None, help="本地数据目录（与 --file 二选一）")
    up.add_argument(
        "--file",
        action="append",
        default=[],
        help="要上传的本地文件路径，可多次指定",
    )
    up.add_argument("--private", action="store_true", help="创建私有仓库")
    up.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="仓库内路径：单文件为最终路径，多文件为目录前缀（如 captions）",
    )
    up.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help=f"仓库 ID，默认 {DEFAULT_REPO_ID}")
    up.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="可选代理地址，例如 http://172.16.5.79:18000（优先于环境变量）",
    )
    up.add_argument(
        "--strict-login-check",
        action="store_true",
        help="严格模式：在线 whoami 校验失败时直接退出（默认仅警告并继续）",
    )

    # ---------- download ----------
    down = subparsers.add_parser("download", help="下载文件或整个仓库")
    down.add_argument(
        "--local-dir",
        type=str,
        default="./data/captions",
        help="保存到本地的目录，默认 ./data/captions",
    )
    down.add_argument(
        "--file",
        action="append",
        default=[],
        help="仓库内文件路径（可多次指定），不指定则下载整个仓库",
    )
    down.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载，覆盖本地已有文件",
    )
    down.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help=f"仓库 ID，默认 {DEFAULT_REPO_ID}")
    down.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="可选代理地址，例如 http://172.16.5.79:18000（优先于环境变量）",
    )
    down.add_argument(
        "--strict-login-check",
        action="store_true",
        help="严格模式：在线 whoami 校验失败时直接退出（默认仅警告并继续）",
    )

    args = parser.parse_args()

    if args.command == "upload":
        if not args.file and not args.dataset_dir:
            raise SystemExit(
                "错误: 必须指定 --file 或 --dataset-dir 之一。"
                "使用 python scripts/data/hf_dataset_sync.py upload --help 查看用法。"
            )
        if args.file and args.dataset_dir:
            raise SystemExit("错误: 不能同时指定 --file 和 --dataset-dir")
        cmd_upload(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

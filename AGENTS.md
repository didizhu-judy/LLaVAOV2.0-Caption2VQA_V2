# Agent 说明

## Hugging Face 数据集 OV2-VideoQA/captions

- **默认 HF 数据集**：`OV2-VideoQA/captions`（用户说「上传到 HF / 从 HF 下载」即指此仓库）。
- **上传**：`python scripts/data/hf_dataset_sync.py upload --file <本地路径>` 或 `--dataset-dir <目录>`。
- **下载**：`python scripts/data/hf_dataset_sync.py download --local-dir <本地目录>` 或 `--file <仓库内文件名>`。
- 详细用法见 `.cursor/rules/hf-captions-sync.mdc`。

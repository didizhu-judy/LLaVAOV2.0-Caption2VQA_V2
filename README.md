# LLaVAOV2.0-Caption2VQA V2 Pipeline

统一的 Ray + asyncio 数据处理流水线，支持：
- `caption_to_vqa`
- `clean_mm_qa`

核心能力：
- `dispatcher + workers + sink` 架构
- 多 worker + 单 worker 内异步并发
- 结果/错误 JSONL 周期落盘
- 基于 `id` 的 resume
- 多后端多端点路由（本地 SGLang/vLLM、多 Azure/OpenAI 端点）

## 目录结构

```text
pipeline/
  core/
    config.py dispatcher.py sink.py worker.py routing.py main.py
  providers/
    base.py openai_compatible.py azure_openai.py registry.py
  tasks/
    base.py
    caption_to_vqa/
      plugin.py prompts.py parser.py adapter.py
    clean_mm_qa/
      plugin.py prompts.py parser.py splitter.py

scripts/
  api/
    start_api_server.sh
    start_api_server_multi.sh
    setup_multi_port_forward.sh
    stop_multi_port_forward.sh
  tasks/
    run_caption_to_vqa.sh
    run_clean_pipeline.sh
  workflow/
    workflow_start_server_and_forward.sh
  env/
    endpoints.json
    local.env.example
    vllm_model.env

configs/
  caption_to_vqa.yaml
  clean_mm_qa.yaml
```

## 运行方式

### 1) 统一 Python 入口

```bash
python -m pipeline.core.main --config configs/caption_to_vqa.yaml --task caption_to_vqa
python -m pipeline.core.main --config configs/clean_mm_qa.yaml --task clean_mm_qa
```

### 2) 任务脚本入口（推荐）

```bash
INPUT=/path/to/captions.jsonl BACKEND_PROFILE=local_multi bash scripts/tasks/run_caption_to_vqa.sh
INPUT=/path/to/mmqa.jsonl BACKEND_PROFILE=azure_multi bash scripts/tasks/run_clean_pipeline.sh
```

`BACKEND_PROFILE` 支持：
- `local_multi`
- `azure_multi`
- `openai_multi`

### 3) 本地/集群 API 启动

```bash
# 单实例
bash scripts/api/start_api_server.sh --backend sglang --port 10025

# 多实例
NUM_SGLANG_INSTANCES=8 bash scripts/api/start_api_server_multi.sh
```

启动成功后会生成：
- `runtime/endpoints.local.json`

任务脚本在 `BACKEND_PROFILE=local_multi` 时会优先读取该文件。

### 4) 一键完整流程（SGLang）

```bash
INPUT=/path/to/captions.jsonl bash scripts/workflow/run_full_pipeline_sglang.sh
```

## 路由策略

配置字段：
- `route_strategy`: `stable_hash` | `least_inflight_weighted`
- `route_failover`: `rotate_on_retry` | `same_endpoint`

默认：
- `stable_hash + rotate_on_retry`

## 端点配置

默认端点文件：`scripts/env/endpoints.json`

支持字段：
- `name`, `provider`, `url`, `model`, `auth_type`, `api_key`, `api_key_env`
- `api_version`, `deployment`, `scope`
- `max_concurrent`, `weight`, `timeout_sec`, `extra_headers`

provider：
- `openai_compatible`
- `azure_openai`

## 测试

```bash
pytest -q
```

如果环境里没有 `pytest`，先安装：

```bash
pip install -e .[dev]
```

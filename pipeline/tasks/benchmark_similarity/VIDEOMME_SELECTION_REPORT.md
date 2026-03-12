# VideoMME 数据筛选报告

最后更新：2026-03-11

## 1. 问题定义

最终目标不是泛化地“提升视频能力”，而是更具体地：

- 从 `Molmo2` 里挑出**最可能提升 `VideoMME`** 的训练数据
- 先服务 `VideoMME`
- 暂时不把 `LVBench / LongVideoBench` 作为主优化目标

这一阶段的核心思路是：

1. 先从真实 `VideoMME` 结果里找当前模型最缺的能力
2. 再去 `Molmo2` 里找最可能训练这些能力的数据
3. 不是直接全局做 QA 相似度 top-k

## 2. 已有证据

这一节只做一件事：把已经存在的结果文件、训练数据构成、错误分析报告压成可决策的事实表。  
它回答三个问题：

1. `2B` 那次 SFT 到底发生了什么
2. 当前 `4B` 模型在 `VideoMME` 上到底差在哪些 bucket
3. 这些事实对后续 `Molmo2` 筛数意味着什么

### 2.1 2B 训练前后结果

相关文件：

| 对象 | 文件 |
| --- | --- |
| `2B baseline results` | `/ov2/feilong/lmms-eval-ov2/eval_log/llava_ov2/LLaVA-OneVision-2-2B-32_frames_llava_next_video_2M_10000__/20260311_122233_results.json` |
| `2B baseline samples` | `/ov2/feilong/lmms-eval-ov2/eval_log/llava_ov2/LLaVA-OneVision-2-2B-32_frames_llava_next_video_2M_10000__/20260311_122233_samples_videomme.jsonl` |
| `2B SFT results` | `/ov2/feilong/lmms-eval-ov2/eval_log/llava_ov2/convert__Molmo2_60_180s_200k_2000/20260306_165435_results.json` |
| `2B SFT samples` | `/ov2/feilong/lmms-eval-ov2/eval_log/llava_ov2/convert__Molmo2_60_180s_200k_2000/20260306_165435_samples_videomme.jsonl` |
| `2B SFT data` | `/ov2/feilong/convert_json/Molmo2-videoforsft/Molmo2-videoforsft_0_60s.jsonl` |

#### 2B 总览

| 模型 | 配置 | VideoMME | 说明 |
| --- | --- | ---: | --- |
| `2B baseline` | `fixed_num_frames=64` | `56.04` | 2M mix 训练参考线 |
| `2B SFT 后` | `fixed_num_frames=16` | `53.00` | 继续训练 `Molmo2-videoforsft_0_60s.jsonl` 后结果 |
| `差值` |  | `-3.04` | 总分下降 |

这里有一个必须显式写出的干扰项：

| 干扰项 | baseline | SFT 后 | 影响 |
| --- | --- | --- | --- |
| `fixed_num_frames` | `64` | `16` | 这会放大时序/计数退化，但不能解释为什么 `Temporal Reasoning` 反而上涨 |
| 训练分布 | 2M mix | `0_60s` Molmo2 子集 | 这是更主要的分布变化来源 |

结论：

- `2B` 结果不是单纯的 frame 数问题
- 如果只是 frame 数变小，最自然的结果应该是时序类一起变差
- 但实际看到的是 `Temporal Reasoning` 上涨、其他通用任务下滑，说明**数据构成确实改变了模型能力方向**

#### 2B 按时长结果

| 时长 | 2B SFT 后 | 2B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `short` | `64.33` | `70.33` | `-6.00` |
| `medium` | `50.22` | `52.67` | `-2.45` |
| `long` | `44.44` | `45.11` | `-0.67` |

这张表的实际含义：

| 观察 | 解释 |
| --- | --- |
| `short` 掉最多 | 这次 SFT 不是“专门没补到长视频”，而是损伤了原来的通用短视频能力 |
| `long` 只小幅变化 | 说明没有真正引入有效的长视频主力训练信号 |
| `medium/long` 都没有被拉起 | `0_60s` 训练分布和 `LongCapQA = 0` 是核心原因 |

#### 2B 按任务结果

| 任务 | 2B SFT 后 | 2B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `Temporal Reasoning` | `41.81` | `32.20` | `+9.61` |
| `Spatial Perception` | `64.81` | `61.11` | `+3.70` |
| `Temporal Perception` | `58.18` | `58.18` | `0.00` |
| `Object Reasoning` | `48.90` | `50.22` | `-1.32` |
| `Information Synopsis` | `70.59` | `73.68` | `-3.09` |
| `Action Reasoning` | `42.81` | `46.32` | `-3.51` |
| `Attribute Perception` | `68.92` | `72.52` | `-3.60` |
| `Counting Problem` | `36.57` | `41.79` | `-5.22` |
| `Spatial Reasoning` | `67.86` | `73.21` | `-5.35` |
| `OCR Problems` | `56.12` | `61.87` | `-5.75` |
| `Action Recognition` | `48.56` | `54.63` | `-6.07` |
| `Object Recognition` | `56.21` | `62.71` | `-6.50` |

#### 2B 逐题净变化

这张表比 aggregate score 更接近“训练数据到底在推模型往哪里走”。

| 任务 | 新增答对 | 新增答错 | 净变化 |
| --- | ---: | ---: | ---: |
| `Temporal Reasoning` | `31` | `14` | `+17` |
| `Spatial Perception` | `5` | `3` | `+2` |
| `Temporal Perception` | `6` | `6` | `0` |
| `Spatial Reasoning` | `2` | `5` | `-3` |
| `Object Reasoning` | `47` | `53` | `-6` |
| `Attribute Perception` | `15` | `23` | `-8` |
| `OCR Problems` | `11` | `19` | `-8` |
| `Information Synopsis` | `26` | `36` | `-10` |
| `Action Reasoning` | `25` | `35` | `-10` |
| `Counting Problem` | `25` | `39` | `-14` |
| `Action Recognition` | `33` | `52` | `-19` |
| `Object Recognition` | `27` | `50` | `-23` |

这张表说明的不是“什么没学到”，而是：

- 训练后模型**确实学到了一部分东西**，否则不会出现 `Temporal Reasoning +17`
- 但它同时丢掉了更多通用识别、OCR、counting 和 recognition 能力
- 所以问题是**混合方式错误**，不是“Molmo2 完全没用”

#### 2B 按内容大类结果

| 类别 | 2B SFT 后 | 2B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `Sports Competition` | `52.67` | `51.33` | `+1.34` |
| `Artistic Performance` | `57.50` | `58.61` | `-1.11` |
| `Knowledge` | `52.96` | `55.19` | `-2.23` |
| `Multilingual` | `50.00` | `53.33` | `-3.33` |
| `Film & Television` | `57.50` | `61.94` | `-4.44` |
| `Life Record` | `48.57` | `56.03` | `-7.46` |

这对筛数的意义：

| 观察 | 数据筛选含义 |
| --- | --- |
| `Life Record` 掉最多 | 不能继续用大量局部 counting / 开放式 QA 去覆盖生活类视频理解 |
| `Sports Competition` 略有收益 | 可能有少量运动时序/动作样本产生了局部正迁移 |
| `Film & Television` 下降明显 | 开放式 SFT 格式和 benchmark-style MCQ 不对齐会伤这类任务 |

### 2.2 2B 训练数据构成与复盘

#### 2B SFT 训练数据来源构成

| source | 行数 | 占比 | 唯一 video id | 平均每视频样本数 |
| --- | ---: | ---: | ---: | ---: |
| `VideoCountEval` | `129,911` | `44.36%` | `129,881` | `1.00` |
| `VideoCapQA` | `69,439` | `23.71%` | `69,439` | `1.00` |
| `AskModelAnything` | `55,917` | `19.09%` | `18,658` | `3.00` |
| `Cap` | `27,242` | `9.30%` | `27,240` | `1.00` |
| `VideoSubtitleQA` | `10,046` | `3.43%` | `10,046` | `1.00` |
| `CapEval` | `314` | `0.11%` | `314` | `1.00` |

#### 2B 训练分布的关键事实

| 事实 | 证据 | 对 `VideoMME` 的含义 |
| --- | --- | --- |
| 训练集明显偏短视频 | 文件名直接是 `0_60s` | 无法系统性补 `medium/long` |
| 没有 `LongCapQA` | `0 条` | 缺少长视频顺序、全局上下文主力 |
| `VideoSubtitleQA` 很少 | `3.43%` | 字幕-视频桥接信号太弱 |
| `VideoCountEval` 占比过高 | `44.36%` | 容易把训练分布拉向局部 counting / grounding |
| `AskModelAnything` 占比高且重复高 | `19.09%`，平均每视频 `3` 条 | 会放大开放式风格和局部模板偏置 |

#### 2B 时序监督来源证据

从 SFT `jsonl` 回表后，时序相关 QA 量如下：

| source | 总 QA | 时序相关 QA | 占比 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `VideoSubtitleQA` | `42,733` | `42,733` | `100.0%` | 量不大，但和 `VideoMME` 的“视频+字幕”模式最对齐 |
| `Cap` | `225,265` | `198,018` | `87.9%` | 大量时间片段描述监督 |
| `VideoCapQA` | `345,142` | `134,786` | `39.1%` | 结构化时序监督主力 |
| `AskModelAnything` | `55,917` | `9,703` | `17.4%` | 有少量时序，但风格更散 |
| `VideoCountEval` | `303,205` | `5,052` | `1.7%` | 基本不是时序主力 |

#### 2B 最可信的局部有效来源

| source | 证据 | 为什么判断它有效 |
| --- | --- | --- |
| `VideoCapQA` | `scene sequence / event sequence / action sequence / process description / event causality / temporal reasoning` | 直接对应 `VideoMME Temporal Reasoning` 的顺序、流程、因果 |
| `VideoSubtitleQA` | `Temporal Sequence Bridging / Forward Alignment / Reverse Alignment / Explanation Grounding / Cross-modal Reasoning` | 直接训练字幕-画面时间桥接，与 `VideoMME` 输入模式高度一致 |
| `Cap` | `what happens between / at / after / before` | 提供时间片段描述，能补过程与时间锚点表征 |

#### 2B 最可疑的拖后腿来源

| source / 因素 | 问题 | 为什么可疑 |
| --- | --- | --- |
| `VideoCountEval` 高比例混入 | `44.36%` | 其内部并不等于纯 benchmark-style counting，混有大量 grounding / reference |
| `AskModelAnything` 高比例混入 | `19.09%` | 开放式长答案、多轮风格与 `VideoMME` 的单轮 MCQ 格式不对齐 |
| `0_60s` 分布 | 全局约束 | 先天不支撑长视频 bucket |
| `CapEval` 混入 | `0.11%` | 量很小，但属于 eval 性质样本，不应进入训练池 |

#### 2B 复盘结论

| 现象 | 直接证据 | 结论 |
| --- | --- | --- |
| `Temporal Reasoning` 上涨 | 分数 `+9.61`，逐题净变化 `+17` | 训练集中确实有一部分时序监督有效 |
| `overall` 下滑 | `56.04 -> 53.00` | 有效监督被大量失配数据稀释 |
| `Counting / OCR / Recognition` 受损 | 多项任务净损失显著 | 训练分布破坏了原有通用识别与文本感知能力 |
| `short` 掉最多 | `-6.00` | 这次 SFT 不只是“长视频没补到”，而是引入了明显的 drift / forgetting |
| `long` 没涨 | `44.44 -> 45.11` 基本不变 | 没有引入真正的长视频主力数据 |

### 2.3 4B 当前模型与 4B baseline

相关文件：

| 对象 | 文件 |
| --- | --- |
| `4B current results` | `/ov2/xiangan/lmms-eval-ov2/eval_log/llava_onevision2_chat/ax_instruct_4b_add180s_add10mins_4b_v2__iter_0004200_hf/20260311_154106_results.json` |
| `4B current samples` | `/ov2/xiangan/lmms-eval-ov2/eval_log/llava_onevision2_chat/ax_instruct_4b_add180s_add10mins_4b_v2__iter_0004200_hf/20260311_154106_samples_videomme.jsonl` |
| `4B baseline results` | `/ov2/feilong/lmms-eval/eval_log/qwen3_vl/convert__Qwen3_vl_4B/20260308_115646_results.json` |
| `4B baseline samples` | `/ov2/feilong/lmms-eval/eval_log/qwen3_vl/convert__Qwen3_vl_4B/20260308_115646_samples_videomme.jsonl` |

#### 4B 总览

| 模型 | 配置 | VideoMME |
| --- | --- | ---: |
| `4B 当前模型` | `fixed_num_frames=120` | `62.63` |
| `4B baseline` | `max_num_frames=256` | `67.26` |
| `差值` |  | `-4.63` |

#### 4B 按时长结果

| 时长 | 4B 当前 | 4B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `short` | `75.78` | `77.44` | `-1.66` |
| `medium` | `60.11` | `67.22` | `-7.11` |
| `long` | `52.00` | `57.11` | `-5.11` |

#### 4B 按任务结果

| 任务 | 4B 当前 | 4B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `Spatial Reasoning` | `82.14` | `76.79` | `+5.35` |
| `Action Reasoning` | `58.60` | `55.79` | `+2.81` |
| `Attribute Perception` | `76.58` | `78.38` | `-1.80` |
| `Action Recognition` | `62.62` | `65.18` | `-2.56` |
| `Object Reasoning` | `57.49` | `62.78` | `-5.29` |
| `Spatial Perception` | `68.52` | `74.07` | `-5.55` |
| `Counting Problem` | `45.52` | `51.12` | `-5.60` |
| `Information Synopsis` | `75.85` | `81.42` | `-5.57` |
| `Object Recognition` | `68.93` | `74.86` | `-5.93` |
| `Temporal Perception` | `65.45` | `74.55` | `-9.10` |
| `OCR Problems` | `67.63` | `76.98` | `-9.35` |
| `Temporal Reasoning` | `41.24` | `55.37` | `-14.13` |

#### 4B 按内容大类结果

| 类别 | 4B 当前 | 4B baseline | 差值 |
| --- | ---: | ---: | ---: |
| `Artistic Performance` | `69.44` | `68.33` | `+1.11` |
| `Film & Television` | `67.50` | `69.72` | `-2.22` |
| `Multilingual` | `58.89` | `62.22` | `-3.33` |
| `Knowledge` | `62.84` | `68.02` | `-5.18` |
| `Life Record` | `60.63` | `66.67` | `-6.04` |
| `Sports Competition` | `56.44` | `64.89` | `-8.45` |

#### 4B 当前最优先修复的 bucket

这张表不是 PDF 抄写，而是直接根据当前 `4B samples` 和 `4B baseline samples` 聚合 `(duration, task_type)` 得到的优先级。

| rank | bucket | wrong_count | error_rate | baseline_gap_pp | priority_score | tier |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `1` | `long × Object Reasoning` | `116` | `48.33%` | `6.25` | `188.50` | `critical` |
| `2` | `long × Temporal Reasoning` | `61` | `67.03%` | `17.58` | `168.24` | `critical` |
| `3` | `medium × Counting Problem` | `60` | `63.16%` | `9.48` | `116.88` | `critical` |
| `4` | `long × Information Synopsis` | `55` | `33.74%` | `9.20` | `105.60` | `critical` |
| `5` | `medium × Object Reasoning` | `55` | `41.04%` | `6.71` | `91.91` | `critical` |
| `6` | `medium × Temporal Reasoning` | `40` | `54.79%` | `12.32` | `89.28` | `hard` |
| `7` | `medium × Action Recognition` | `52` | `43.70%` | `6.73` | `87.00` | `hard` |
| `8` | `long × Action Reasoning` | `84` | `46.67%` | `-2.22` | `84.00` | `hard` |
| `9` | `medium × Object Recognition` | `41` | `31.06%` | `9.09` | `78.27` | `hard` |
| `10` | `medium × OCR Problems` | `30` | `44.12%` | `13.24` | `69.72` | `hard` |
| `11` | `long × Counting Problem` | `30` | `62.50%` | `10.42` | `61.26` | `hard` |
| `12` | `short × Counting Problem` | `56` | `44.80%` | `0.80` | `60.48` | `hard` |

这张表对后续筛数的约束很明确：

| 事实 | 数据筛选含义 |
| --- | --- |
| `long × Object Reasoning` 排第 1 | 不能只盯时序，必须把长视频 object reasoning 当作头号目标 |
| `long × Temporal Reasoning` 排第 2 | `VideoCapQA / VideoSubtitleQA / LongCapQA` 应该是第一梯队 |
| `medium × Counting Problem` 排第 3 | `VideoCountEval` 不能全砍，但必须过滤成纯 counting |
| `long × Information Synopsis` 进入前 4 | 长视频 summary / 全局理解数据也要进入第一轮，不应被只看 temporal 的方案忽略 |
| `medium × OCR Problems` 进入前 10 | 后续筛数不能只补时序和 counting，否则会继续丢 OCR |

### 2.4 现有 PDF 报告的作用

相关文件：

- `/root/LLaVAOV2.0-Caption2VQA_V2/output/VideoMME 错误分析详细报告.pdf`

#### PDF 已经提供的内容

| 维度 | PDF 是否覆盖 | 作用 |
| --- | --- | --- |
| `overall / duration / task_type / sub_category` | 是 | 给出当前 4B 的完整错误画像 |
| `时长 × 任务` 优先级矩阵 | 是 | 直接指出哪些 bucket 值得优先修 |
| `option bias` | 是 | 提醒存在 `A` 选项偏差 |
| `典型 long + temporal` 错例 | 是 | 说明长视频时序追踪失败的模式 |

#### PDF 没有覆盖、但筛数必须补上的内容

| 缺口 | 为什么重要 |
| --- | --- |
| `2B 训练前后结果` | 只有看到历史正负迁移，才能知道哪些 `Molmo2` 来源曾经有效 |
| `SFT 训练数据构成` | 不知道混了什么数据，就无法解释为什么会涨一项、掉多项 |
| `Molmo2 数据源结构` | 只有知道 source 内部类型，才能把错误分析变成筛数方案 |
| `source -> benchmark 弱点` 的映射 | 错误分析只能告诉我们“缺什么”，筛数还需要知道“拿什么补” |

#### PDF 和真实结果结合后的结论

| 结论 | 依据 |
| --- | --- |
| PDF 适合做当前 4B 错误诊断 | 它覆盖了 `duration/task/sub_category/option bias` |
| PDF 不足以直接指导数据筛选 | 它不包含 `2B` 历史、SFT 数据构成、Molmo2 source 结构 |
| 当前筛数主线应以真实结果文件为准 | `priority bucket`、`2B 正负迁移`、`训练集来源构成` 都需要从真实文件直接算 |

## 3. 从第一性原理得到的筛数逻辑

### 3.1 为什么不能直接做全局 QA 相似度 top-k

因为我们真正要解决的是：

- 当前模型在 `VideoMME` 的哪些能力最弱
- 哪些 `Molmo2` 数据真的在训练这些能力

而不是：

- 哪些题表面上和 benchmark 题最像

仅做全局 QA 相似度会有三个问题：

1. 容易把“文本相似但能力不匹配”的数据选进来
2. 容易让数据量大的来源主导训练集
3. 无法优先服务当前最痛的 `VideoMME` 弱点

### 3.2 正确顺序

更合理的顺序是：

1. `VideoMME` 先找弱点桶
2. `Molmo2` 再做能力标签
3. 在同类能力里做相似度排序
4. 最后再做 source cap 和抽样

也就是：

- benchmark 决定“先补哪里”
- Molmo2 决定“拿什么补”

## 4. 最小标签方案

### 4.1 为什么保留 `difficulty_tier`

`difficulty_tier` 只给 `VideoMME` 的弱点桶用。  
它不需要调 API，也不需要额外标注。

它的作用只是：

- 把当前最值得优先补的 bucket 排出来

这一阶段只按 `(duration, task_type)` 聚合，并计算：

- `wrong_count`
- `error_rate`
- `baseline_gap_pp`
- `priority_score`
- `difficulty_tier`

### 4.2 为什么保留 `skill_bucket`

`skill_bucket` 只给 `Molmo2` 候选数据用。  
它回答的是：

- 这条数据主要在训练什么能力

当前只保留最小的 8 类：

- `temporal_sequence`
- `counting`
- `object_reasoning`
- `action_reasoning`
- `subtitle_alignment`
- `summary`
- `ocr_text`
- `general`

这足够支撑第一轮 `VideoMME` 定向筛数。

### 4.3 为什么暂不做更复杂标签

当前不把下面这些作为主方案：

- 复杂 `failure_mode`
- 大量 `format_tags`
- `quality_risk_tags`
- 复杂 `difficulty_proxy_tags`

原因很简单：

- 第一阶段目标不是做最精细的 taxonomy
- 第一阶段目标是先把筛数逻辑跑通

所以现在只保留：

- benchmark 侧：`difficulty_tier`
- candidate 侧：`source_family`、`skill_bucket`、`duration_hint`

## 5. 第一轮筛数方案

### 5.1 `VideoMME` 先找弱点桶

当前 selector 已经覆盖 6 个 bucket：

1. `long × Object Reasoning`
2. `long × Action Reasoning`
3. `long × Temporal Reasoning`
4. `medium × Counting Problem`
5. `long × Information Synopsis`
6. `medium × OCR Problems`

这 6 个桶覆盖了当前最值得补的：

- 长视频 object / action / temporal / summary
- 中视频 counting / OCR

### 5.2 Molmo2 的能力标签映射

主来源及默认理解：

| source | 第一轮主要价值 |
| --- | --- |
| `VideoCapQA` | 时序顺序、过程、Object/Action reasoning |
| `VideoSubtitleQA` | 字幕-画面对齐、时序桥接、文本相关推理 |
| `LongCapQA` | 长视频上下文、长链条顺序 |
| `Cap` | 时间片段描述、流程型监督 |
| `AskModelAnything` | 少量时序或因果补充 |
| `VideoCountEval` | 纯计数类补充 |

### 5.3 第一轮过滤规则

固定过滤：

- `CapEval` 全排除
- `VideoCountEval` 只保留：
  - `object`
  - `action/event`
  - `animal`
- `VideoCountEval` 排除：
  - `comparative reference`
  - `referring expression`
  - `indirect reference`
  - `spatial reference`
  - `anomaly`
- `AskModelAnything` 中 `skill_bucket = general` 的样本排除

### 5.4 第一轮选择逻辑

不是直接全局 top-k，而是：

1. 每个弱点桶先确定允许的 `skill_bucket`
2. 只在允许集合里选候选
3. 桶内先做 `exact skill` 优先级
4. 再按：
   - `semantic similarity`
   - `skill_bucket priority`
   - `duration match`
   排序
5. 最后做 `source priority + soft source cap`

固定映射：

- `Temporal Reasoning` -> `{temporal_sequence, subtitle_alignment}`
- `Object Reasoning` -> `{object_reasoning, subtitle_alignment}`
- `Action Reasoning` -> `{action_reasoning, temporal_sequence, subtitle_alignment}`
- `Counting Problem` -> `{counting}`
- `Information Synopsis` -> `{summary, temporal_sequence, subtitle_alignment}`
- `OCR Problems` -> `{ocr_text, subtitle_alignment}`

当前实现里的额外约束：

| bucket | primary source | secondary source |
| --- | --- | --- |
| `long × Temporal Reasoning` | `LongCapQA / VideoCapQA / VideoSubtitleQA` | `Cap / AskModelAnything` |
| `long × Object Reasoning` | `LongCapQA / VideoCapQA / VideoSubtitleQA` | `Cap / AskModelAnything` |
| `long × Action Reasoning` | `LongCapQA / VideoCapQA / VideoSubtitleQA` | `Cap / AskModelAnything` |
| `medium × Counting Problem` | `VideoCountEval` | `VideoCapQA / AskModelAnything` |
| `long × Information Synopsis` | `LongCapQA / Cap / VideoCapQA` | `VideoSubtitleQA / AskModelAnything` |
| `medium × OCR Problems` | `VideoSubtitleQA / VideoCapQA` | `Cap / AskModelAnything` |

当前实现里的 bucket 内 `skill_bucket` 优先级：

| bucket | skill priority |
| --- | --- |
| `Temporal Reasoning` | `temporal_sequence > subtitle_alignment` |
| `Object Reasoning` | `object_reasoning > subtitle_alignment` |
| `Action Reasoning` | `action_reasoning > subtitle_alignment > temporal_sequence` |
| `Counting Problem` | `counting` only |
| `Information Synopsis` | `summary > temporal_sequence > subtitle_alignment` |
| `OCR Problems` | `ocr_text > subtitle_alignment` |

### 5.5 第一轮 demo 范围

demo 只做真实小切片：

- 使用真实 `4B` 当前结果
- 使用真实 `4B baseline`
- 每个 source 最多预取 `5000`
- 每个 bucket 导出 `top 50`
- 合并导出 `top 200`

输出目录：

- `output/videomme_selector_demo/`

### 5.6 当前 demo 输出与反思

当前真实 demo 的输出文件已经落到：

- `output/videomme_selector_demo/priority_buckets.json`
- `output/videomme_selector_demo/candidate_label_summary.json`
- `output/videomme_selector_demo/bucket_sampling_plan.json`
- `output/videomme_selector_demo/bucket_long_temporal_reasoning_top50.jsonl`
- `output/videomme_selector_demo/bucket_long_object_reasoning_top50.jsonl`
- `output/videomme_selector_demo/bucket_long_action_reasoning_top50.jsonl`
- `output/videomme_selector_demo/bucket_medium_counting_problem_top50.jsonl`
- `output/videomme_selector_demo/bucket_long_information_synopsis_top50.jsonl`
- `output/videomme_selector_demo/bucket_medium_ocr_problems_top50.jsonl`
- `output/videomme_selector_demo/merged_top200.jsonl`

#### 当前 demo 的 bucket 结果

| bucket | source_family 分布 | skill_bucket 分布 | 判断 |
| --- | --- | --- | --- |
| `long × Temporal Reasoning` | `longcapqa 17`, `capqa 17`, `subtitleqa 16` | `temporal_sequence 34`, `subtitle_alignment 16` | 合理，已经回到长时序主力 source |
| `long × Object Reasoning` | `longcapqa 17`, `capqa 17`, `subtitleqa 16` | `object_reasoning 34`, `subtitle_alignment 16` | 合理，object reasoning 不再被通用 QA 淹没 |
| `long × Action Reasoning` | `longcapqa 17`, `capqa 17`, `subtitleqa 16` | `action_reasoning 18`, `temporal_sequence 16`, `subtitle_alignment 16` | 比第一版明显更好，但仍有一部分 temporal 代理样本 |
| `medium × Counting Problem` | `count_eval 50` | `counting 50` | 合理，已经变成纯 counting 桶 |
| `long × Information Synopsis` | `longcapqa 17`, `caption 17`, `capqa 16` | `summary 49`, `temporal_sequence 1` | 合理，summary 桶已经有自己的 source 结构 |
| `medium × OCR Problems` | `subtitleqa 50` | `ocr_text 36`, `subtitle_alignment 14` | 合理，OCR 桶已经能拿到显式文本类样本 |

#### 当前 demo 的合并结果

| 指标 | 数值 |
| --- | ---: |
| `merged_top_count` | `200` |
| `longcapqa` | `64` |
| `capqa` | `51` |
| `subtitleqa` | `50` |
| `count_eval` | `18` |
| `caption` | `17` |
| `askmodelanything` | `0` |


#### 当前 demo 的采样配额草案

这个文件不是最终训练配比，而是把 `priority_score` 转成相对 bucket 权重，作为后续配额设计的起点：

- `output/videomme_selector_demo/bucket_sampling_plan.json`

当前 6 个 bucket 的相对权重如下：

| bucket | difficulty_tier | bucket_weight |
| --- | --- | ---: |
| `long × Object Reasoning` | `critical` | `0.2572` |
| `long × Temporal Reasoning` | `critical` | `0.2295` |
| `medium × Counting Problem` | `critical` | `0.1595` |
| `long × Information Synopsis` | `critical` | `0.1441` |
| `long × Action Reasoning` | `hard` | `0.1146` |
| `medium × OCR Problems` | `hard` | `0.0951` |

这张表的作用只有一个：

- 先给训练数据构造一个按弱点排序的相对预算
- 暂时不把它当成最终全局 mixture 比例

这里要明确一个 caveat：

| 现象 | 原因 |
| --- | --- |
| `capqa` 的 `duration_hint` 大量仍是 `unknown` | 当前 metadata 不足以精确恢复时长，这会影响 long bucket 的细粒度排序 |
| 当前 demo 使用的是 `token_overlap` | 它适合先验证标签和过滤逻辑，不代表最终检索 backend 已经最优 |

#### 这轮 demo 说明了什么

| 观察 | 解释 | 后续含义 |
| --- | --- | --- |
| `LongCapQA` 已经进入三个 long reasoning 桶 | 修正 `LongCapQA` 的 `Category` 透传后，selector 能识别它的长视频能力标签 | `LongCapQA` 应进入第一梯队 |
| `AskModelAnything` 从 top 输出里消失 | 收紧时序关键词并把它降为 secondary source 后，噪声被压下去了 | `AskModelAnything` 只能作为补充，不应作为主力 |
| `medium × Counting` 变成纯 `count_eval` | 过滤掉 `reference/grounding` 类后，counting 桶终于和 `VideoMME Counting Problem` 对齐 | `VideoCountEval` 不是不能用，而是必须先做类别过滤 |
| `long × Information Synopsis` 会稳定选出 `LongCapQA / Cap / CapQA` | 这些 source 里确实存在 `summary / video topic / event summary` 类样本 | `Information Synopsis` 不应再被时序 bucket 顺带处理，而应独立建桶 |
| `medium × OCR` 会稳定选出 `VideoSubtitleQA` | `text recognition / text location` 在 subtitle QA 里足够多 | OCR 可以独立做针对性筛数，不必混在通用 QA 里 |
| `Action Reasoning` 仍混入一部分 `temporal_sequence` | 说明 `VideoMME Action Reasoning` 和时序顺序类确实有交叠，但 exact action reasoning 还不够多 | 后续可以继续增加 action reasoning 的 exact-match 权重 |
| `merged_top200` 现在能补满 | 加入 summary / OCR 两个 bucket 后，候选空间明显增大 | 当前 selector 已经可以进入 manifest / 配额层，而不只是 demo |

## 6. 备选增强方案

如果第一轮结果表明：

- `skill_bucket` 还不够细
- 某些来源内部差异仍然很大
- 纯关键词和显式 category 还不足以稳定筛数

再考虑第二阶段增强：

1. 更细的时序子标签
   - 例如 `sequence / causality / step-following / time-anchor`
2. 更细的格式风险分析
   - 例如 `open-ended / benchmark-like / multi-turn`
3. 更细的长视频代理标签
   - 例如跨段推理、字幕依赖、全局摘要

但这些都不是第一轮主方案的一部分。

## 7. 下一步该做什么

这一阶段最重要的不是先拍脑袋定全局占比，而是先把下面两件事做对：

1. `VideoMME` 弱点桶分析程序
2. `Molmo2` 最小标签 + 分桶筛数 demo

在这两件事已经跑通后，下一步应直接转向更接近训练集构造的动作：

| 优先级 | 动作 | 目的 |
| ---: | --- | --- |
| `1` | 从 `merged_top200` 扩展到每个 bucket 的真实采样配额表 | 把 selector 输出变成可训练的数据清单 |
| `2` | 把 `count_eval` 的全局 cap 从 demo 规则转成可控配额参数 | 让训练集构造从 demo 过渡到真正的 mixture 控制 |
| `3` | 在 `Action Reasoning` 桶里继续提高 exact `action_reasoning` 权重 | 减少被 `temporal_sequence` 代理样本占位 |
| `4` | 用小规模训练验证 `bucket-aware` 采样是否优于旧的 `0_60s` 混合 | 把当前分析转成真实收益判断 |

后续验证标准：

- `VideoMME overall`
- `short / medium / long`
- `Temporal Reasoning`
- `Counting Problem`
- `Object Reasoning`
- `Action Reasoning`

如果只涨 `Temporal Reasoning`，但 `overall` 和 `Counting / OCR / Object Recognition` 继续掉，说明：

- 这批数据只带来了局部收益
- 但筛数和混合方式仍然不对

这时要继续调：

- source cap
- `VideoCountEval` 权重
- `AskModelAnything` 过滤强度
- `LongCapQA / VideoSubtitleQA / VideoCapQA` 的占比

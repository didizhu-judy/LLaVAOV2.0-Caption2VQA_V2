# Benchmark 画像

这份文档汇总了针对 Molmo2 做数据选择和配比加权时，最有用的 benchmark 侧属性信息。

最后更新：2026-03-10

## 信息来源说明

这份文档融合了两类信息：

- `本地已验证`：来自当前机器上已经下载到缓存的官方本地文件，或来自公开 benchmark 仓库中直接可验证的信息。
- `文献总结 / 工作笔记`：来自论文式总结、讨论笔记和研究归纳的高层 benchmark 定位、规模对比、难点分析与评估建议。

做脚本实现时，应以 `本地已验证` 部分作为事实依据。
做 benchmark 对齐、差距分析和数据选择策略时，可参考 `文献总结 / 工作笔记` 部分。

## 建议的统一 Profiling 维度

为了做跨 benchmark 对比，最值得统一的公共维度是：

- `duration_bucket`
- `video_domain` 或 `video_type`
- `fine_category`
- `task_category`
- `reasoning_level`
- `scope_locality`
- `language_or_subtitle_dependency`

这三个 benchmark 暴露出来的 schema 不完全相同，但都可以对齐到上面的统一坐标系中。

## 跨 Benchmark 定位

### 一句话定位

| benchmark | 核心定位 | 最适合的使用场景 |
| --- | --- | --- |
| `VideoMME` | 覆盖时长、内容领域、任务类型的通用视频 MLLM 综合 benchmark | 做整体能力评估和短板诊断 |
| `LVBench` | 面向超长视频理解的 benchmark，目标是最长 2 小时视频 | 测长时记忆、检索和整段视频理解 |
| `LongVideoBench` | 以引用式推理为核心的长上下文视频-语言交错理解 benchmark | 测长视频+字幕条件下的细粒度引用推理 |

### 实际理解方式

- 如果问题是“模型在各种常见视频场景上的总体视频理解能力怎么样”，优先看 `VideoMME`。
- 如果问题是“模型看完整集 / 完整比赛 / 完整长记录之后，还能不能回忆和检索信息”，优先看 `LVBench`。
- 如果问题是“模型能否在长视频+字幕交错上下文中，正确围绕某个被引用的片段或多段上下文进行推理”，优先看 `LongVideoBench`。

## 跨 Benchmark 规模与时长

### 文献总结 / 工作笔记

下表汇总了研究过程中讨论过的 benchmark 规模信息。这些值适合用于规划数据策略，但如果它们会直接影响脚本逻辑或论文结果，仍然建议用官方发布版本再次核对。

| 属性 | VideoMME | LVBench | LongVideoBench |
| --- | --- | --- | --- |
| 视频数量 | `900` | 约 `103` 条视频 | `3,763` 条视频 |
| QA 数量 | 约 `2,700` 条多选 QA | 约 `1,549` 条 QA | `6,678` 条多选 QA |
| 总时长 | `254` 小时（官方论文明确给出） | 官方 README 未直接给出总时长；可确认单视频最长到 `2` 小时 | 官方 README / abstract 未直接给出总时长；可确认视频最长到约 `1` 小时 |
| 主要来源形态 | 新采集并重新标注 | 来自公开视频平台的长视频，专为长视频 QA 筛选 | 带字幕的网络长视频 |

### 时长设计对比

| benchmark | 时长设计 | 解读 |
| --- | --- | --- |
| `VideoMME` | `short`、`medium`、`long` | 用粗粒度、平衡的时长桶做通用能力报告 |
| `LVBench` | 没有简单 short/medium/long 划分，数据本身就是长视频 | 重点就是极长视频，往往是完整事件或完整节目 |
| `LongVideoBench` | 4 个时长组，从短片段一直到小时级内容 | 比 VideoMME 更细地拆解长上下文难度 |

### 使用建议

- 如果你想要一个干净的 short-vs-medium-vs-long 性能曲线，用 `VideoMME`。
- 如果你想在长上下文场景里看更细的时长分桶，用 `LongVideoBench`。
- 如果研究问题本身就是超长时间跨度理解，用 `LVBench`。

## VideoMME

### 官方本地文件

- `/root/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/README.md`
- `/root/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet`

### 官方字段

- `video_id`
- `duration`
- `domain`
- `sub_category`
- `url`
- `videoID`
- `question_id`
- `task_type`
- `question`
- `options`
- `answer`

### Split 大小

- 总样本数：`2700`

### 时长桶

| 值 | 数量 |
| --- | ---: |
| `short` | 900 |
| `medium` | 900 |
| `long` | 900 |

### 内容大类

| 值 | 数量 |
| --- | ---: |
| `Knowledge` | 810 |
| `Life Record` | 630 |
| `Sports Competition` | 450 |
| `Film & Television` | 360 |
| `Artistic Performance` | 360 |
| `Multilingual` | 90 |

### 子类别

官方 `sub_category` 一共有 `30` 个值，并且每类都是 `90` 条样本：

- `Humanity & History`
- `Literature & Art`
- `Biology & Medicine`
- `Finance & Commerce`
- `Astronomy`
- `Geography`
- `Law`
- `Life Tip`
- `Technology`
- `Animation`
- `Movie & TV Show`
- `Documentary`
- `News Report`
- `Esports`
- `Basketball`
- `Football`
- `Athletics`
- `Other Sports`
- `Stage Play`
- `Magic Show`
- `Variety Show`
- `Acrobatics`
- `Handicraft`
- `Food`
- `Fashion`
- `Daily Life`
- `Travel`
- `Pet & Animal`
- `Exercise`
- `Multilingual`

### 任务类别

| 值 | 数量 |
| --- | ---: |
| `Object Reasoning` | 454 |
| `Object Recognition` | 354 |
| `Information Synopsis` | 323 |
| `Action Recognition` | 313 |
| `Action Reasoning` | 285 |
| `Counting Problem` | 268 |
| `Attribute Perception` | 222 |
| `Temporal Reasoning` | 177 |
| `OCR Problems` | 139 |
| `Spatial Reasoning` | 56 |
| `Temporal Perception` | 55 |
| `Spatial Perception` | 54 |

### 实用备注

- VideoMME 是做层次化 profiling 最干净的 benchmark，因为它显式给出了时长、粗粒度 domain、细粒度 sub-category 和 task type。
- `duration` 和 `sub_category` 本身是平衡设计。
- `task_type` 则明显不平衡，因此做加权时更应该结合“当前性能差距”而不是只看原始样本量。

### 文献总结 / 工作笔记

VideoMME 是做统一评估坐标系时最适合拿来当“外骨骼”的 benchmark：

- 时长轴已经显式存在
- 内容轴已经显式存在
- 子类别轴已经显式存在
- 任务轴已经显式存在

如果你的目标是给 `VideoMME + LVBench + LongVideoBench` 建一套共同分析框架，VideoMME 是最自然的锚点，因为它本身就接近一套完整的多轴诊断报表。

## LVBench

### 官方来源

- 官方 GitHub 仓库：`https://github.com/zai-org/LVBench`
- 官方 README 明确说明 Hugging Face 数据集提供 `video_info.meta.jsonl`

### 官方本地文件

- `/root/.cache/huggingface/hub/datasets--zai-org--LVBench/snapshots/0caedb92002cc268bad486449e551c76f0485670/README.md`
- `/root/.cache/huggingface/hub/datasets--zai-org--LVBench/snapshots/0caedb92002cc268bad486449e551c76f0485670/video_info.meta.jsonl`

### 官方结构

每一行顶层记录对应一条视频：

- 顶层字段：
  - `key`
  - `qa`
  - `type`
  - `video_info`
- 内层 QA 字段：
  - `uid`
  - `question`
  - `answer`
  - `question_type`
  - `time_reference`

### Split 大小

- 视频总数：`103`
- QA 总数：`1549`

### 视频类型

LVBench 顶层显式暴露了 `6` 个官方 `type`：

| 值 | 数量 |
| --- | ---: |
| `selfmedia` | 21 |
| `cartoon` | 18 |
| `live` | 17 |
| `sport` | 17 |
| `tv` | 16 |
| `documentary` | 14 |

### 问题类型

README 中提到的 `six core capabilities`，在实际元数据里表现为 `question_type`。

按 QA 对统计：

| 值 | 数量 |
| --- | ---: |
| `entity recognition` | 677 |
| `event understanding` | 647 |
| `key information retrieval` | 291 |
| `temporal grounding` | 220 |
| `reasoning` | 201 |
| `summarization` | 58 |

### 实用备注

- LVBench 天然是一个两层结构：
  - `type` 可以看作 domain-like 维度
  - `question_type` 可以看作 capability 维度
- `time_reference` 是一个很有用的补充字段，可以进一步转成 temporal locality 信号。
- 相比 VideoMME，LVBench 在内容子类上没那么细，但在长视频 QA 能力类型上更明确。

### 文献总结 / 工作笔记

LVBench 可以理解为“长视频版本的任务能力 benchmark”：

- `type` 扮演的是高层内容题材维度
- `question_type` 扮演的是能力维度

常见的六类长视频能力可理解为：

- `Temporal Grounding`
- `Summarization`
- `Reasoning`
- `Entity Recognition`
- `Event Understanding`
- `Key Information Retrieval`

这些标签更适合理解为“长视频场景下的高层能力”，而不是底层视觉算子。

因此 LVBench 很适合回答下面这类问题：

- 模型能不能在长视频里定位关键时刻？
- 模型能不能对长视频做总结或保留长程记忆？
- 模型看完长视频后，能不能检索出精确信息？

## LongVideoBench

### 官方本地文件

- `/root/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/README.md`

### 官方访问状态

官方标注文件，如 `lvb_val.json`，所在 Hugging Face 数据集目前是 gated：

- 仓库：`longvideobench/LongVideoBench`
- 当前状态：在没有 Hugging Face 登录认证时会返回 `401 Unauthorized`

因此，当前环境里只缓存了官方 README，官方 JSON/parquet 标注尚未下载成功。

### 从官方 README 可以确认的信息

- 任务形式：multiple-choice visual question answering
- 总问题数：`6678`
- 官方类别数：`17 categories`
- 官方 leaderboard 中使用的时长桶：
  - `8s-15s`
  - `15s-60s`
  - `180s-600s`
  - `900s-3600s`

### 基于公开镜像的 validation 画像

由于官方标注 gated，这一部分字段级 profile 来自公开镜像 `Jialuo21/LongVideoBench` 的 validation parquet。字段名与官方 benchmark 一致，但仍应明确视为“镜像推断结果”，而不是直接来自官方 gated 文件。

使用的镜像文件：

- `/tmp/jialuo21_longvideobench/validation.parquet`

### 镜像中观察到的字段

- `video_id`
- `question`
- `question_wo_referring_query`
- `candidates`
- `correct_choice`
- `position`
- `topic_category`
- `question_category`
- `level`
- `id`
- `video_path`
- `subtitle_path`
- `duration_group`
- `starting_timestamp_for_subtitles`
- `duration`
- `view_count`
- `type`

### 镜像 split 大小

- Validation 样本数：`1337`

### Topic 类别

观察到 `10` 个 `topic_category` 值：

| 值 | 数量 |
| --- | ---: |
| `LV-Lifestyle-Life-Vlogs` | 169 |
| `Recreational: MR-Movie-Recaps` | 152 |
| `LT-Lifestyle-Travel-Guides` | 150 |
| `KH-Knowledge-History` | 147 |
| `LC-Lifestyle-Cooking-Recipes` | 146 |
| `KA-Knowledge-Art` | 142 |
| `KG-Knowledge-Geography` | 124 |
| `NP-News-Programs` | 121 |
| `KS-Knowledge-STEM` | 111 |
| `KC-Knowledge-Computer-Science` | 75 |

### 问题类别

观察到 `17` 个 `question_category` 代码：

| 值 | 数量 |
| --- | ---: |
| `SSS` | 97 |
| `E3E` | 94 |
| `S2E` | 93 |
| `S2A` | 88 |
| `O2E` | 87 |
| `TAA` | 82 |
| `SOS` | 81 |
| `T2A` | 79 |
| `T2O` | 76 |
| `T3O` | 74 |
| `TOS` | 73 |
| `T3E` | 73 |
| `SAA` | 72 |
| `S2O` | 72 |
| `O3O` | 66 |
| `E2O` | 65 |
| `T2E` | 65 |

### 推理层级

| 值 | 数量 |
| --- | ---: |
| `L2-Relation` | 712 |
| `L1-Perception` | 625 |

### 时长组

观察到的 `duration_group` 值：

| 值 | 数量 |
| --- | ---: |
| `3600` | 564 |
| `600` | 412 |
| `15` | 189 |
| `60` | 172 |

### Scope 类型

| 值 | 数量 |
| --- | ---: |
| `local` | 1220 |
| `global` | 117 |

### 实用备注

- LongVideoBench 比 VideoMME 多了两个很重要的显式轴：
  - `level`（`L1-Perception` vs `L2-Relation`）
  - `type`（`local` vs `global`）
- 这两个字段对做长上下文 curriculum 设计非常有价值。
- 一旦后续拿到官方 gated 访问权限，第一步应当做官方 `lvb_val.json` 与镜像字段值的一致性校验。

### 文献总结 / 工作笔记

LongVideoBench 更适合被理解成一个“细粒度推理 benchmark”，而不是“内容题材 benchmark”。

它强调的是：

- 长上下文视频-语言交错理解
- 字幕参与推理
- 围绕前文被引用片段的 referring reasoning
- 结构化的问题类型分析

因此它特别适合回答这类问题：

- 模型能不能对被引用的多个片段之间关系进行推理？
- 模型能不能把字幕证据和视觉证据结合起来？
- 性能是否会随着时长和上下文复杂度明显衰减？

`question_category` 这套编码，本质上可以看成一套紧凑的细粒度引用推理模板 taxonomy。

## 内容轴对比

### 文献总结 / 工作笔记

| benchmark | 内容轴风格 | 说明 |
| --- | --- | --- |
| `VideoMME` | 强内容题材 taxonomy | 6 个顶层 domain + 30 个子类 |
| `LVBench` | 中等粒度的长视频题材类型 | 6 个顶层 type，内容覆盖较广，但报告粒度不如 VideoMME 细 |
| `LongVideoBench` | 内容题材弱、任务类型强 | 虽然有 topic category，但 benchmark 更主要按问题结构和推理类型分析 |

### 解释

- `VideoMME` 和 `LVBench` 更适合按“这是什么类型的视频”来分组。
- `LongVideoBench` 更适合按“这个问题要求什么类型的推理”来分组。

这件事会直接影响 Molmo2 数据筛选：

- 对 `VideoMME`，内容题材对齐很重要。
- 对 `LVBench`，长视频题材覆盖很重要。
- 对 `LongVideoBench`，推理模板对齐比表面题材相似更重要。

## 任务与能力轴对比

### VideoMME 风格

VideoMME 混合了低层感知与高层推理：

- 感知类：`Attribute Perception`、`Temporal Perception`、`Spatial Perception`、`Object Recognition`、`Action Recognition`、`OCR Problems`
- 推理类：`Spatial Reasoning`、`Information Synopsis`、`Object Reasoning`、`Action Reasoning`、`Counting Problem`、`Temporal Reasoning`

### LVBench 风格

LVBench 则按“长视频原生能力”来分：

- `Temporal Grounding`
- `Summarization`
- `Reasoning`
- `Entity Recognition`
- `Event Understanding`
- `Key Information Retrieval`

### LongVideoBench 风格

LongVideoBench 使用更原子化的问题 taxonomy：

- `L1-Perception`
- `L2-Relation`
- `17` 个问题类别代码，例如 `S2E`、`S2O`、`E3E`、`SSS`、`TAA`

### 对齐直觉

| 共同能力轴 | VideoMME | LVBench | LongVideoBench |
| --- | --- | --- | --- |
| 低层识别 | object/action/attribute perception | entity recognition、event understanding | 很多 `L1` 类别 |
| 检索 | OCR、object reasoning、temporal perception | key information retrieval、temporal grounding | 偏 local 的 question categories |
| 总结 | information synopsis | summarization | 偏 global 的问题 |
| 高阶推理 | temporal/object/action reasoning | reasoning | `L2-Relation` 类别 |

## 评估与报表风格对比

### 共性

三个 benchmark 本质上都是多选 QA 准确率 benchmark，并且都支持按 bucket 做分组统计。

### 报表风格

| benchmark | 更自然的报表方式 |
| --- | --- |
| `VideoMME` | overall + duration + domain + sub-category + task type |
| `LVBench` | overall + 长视频能力组 + 视频题材组 |
| `LongVideoBench` | overall + duration group + question category + reasoning level + local/global |

### 推荐做法

如果你想做一个统一的评估面板：

- 用 `VideoMME` 作为外层报表框架
- 把 `LVBench` 作为极长视频能力探针
- 把 `LongVideoBench` 作为细粒度长上下文推理探针

## 共同难点

### 文献总结 / 工作笔记

这三个 benchmark 共同指向了一组相当稳定的困难模式：

- 长上下文推理
- 时序推理
- 计数和数字级精确信息检索
- 长时间跨度事件追踪
- 体育类或快节奏动作视频
- 基于字幕的检索与跨模态推理

### VideoMME 式短板模式

从前面讨论过的 VideoMME 结果图来看，典型弱项包括：

- `Counting Problem`
- `Temporal Reasoning`
- 某些快节奏体育子类，如 `Basketball`

### 跨 benchmark 含义

这些问题并不只存在于 VideoMME：

- `LVBench` 更强调长时程版本的 retrieval、grounding 和 summarization
- `LongVideoBench` 更强调长时程 relational reasoning 与 referred-context understanding

所以，Molmo2 数据选择策略应优先覆盖：

- 长时长样本
- 多事件叙事
- 带字幕 QA
- 显式 temporal reasoning 模式
- 计数和数量敏感问题

## 推荐的统一评估骨架

### 文献总结 / 工作笔记

如果你想把三个 benchmark 放进一个共同分析面板，最干净的结构是：

1. `Duration`
   - short
   - medium
   - long
   - extreme long
2. `Content`
   - knowledge
   - film or TV
   - sports
   - lifestyle or life record
   - documentary or news
   - multilingual or subtitle-heavy
3. `Task`
   - perception
   - retrieval
   - summarization
   - temporal reasoning
   - object or action reasoning
   - referring reasoning
4. `Context complexity`
   - local
   - multi-segment
   - global
   - long-context relational

### 为什么这套骨架有用

这套统一骨架可以让你：

- 在同一坐标系下比较三个 benchmark
- 按共同短板给 Molmo2 分配采样配额
- 区分“通用视频 IQ 提升”和“长上下文推理提升”
- 做更稳定的 ablation 实验

## 跨 Benchmark 对齐总结

| 统一轴 | VideoMME | LVBench | LongVideoBench |
| --- | --- | --- | --- |
| `duration_bucket` | `duration` | 可由 `video_info.duration_minutes` 或 temporal span 推导 | `duration_group` |
| `video_domain` | `domain` | `type` | `topic_category` |
| `fine_category` | `sub_category` | 无显式细粒度字段，主要靠 `type` | `topic_category` |
| `task_category` | `task_type` | `question_type` | `question_category` |
| `reasoning_level` | 隐含在 `task_type` 中 | 隐含在 `question_type` 中 | 显式 `level` |
| `scope_locality` | 隐式 | 可由 `time_reference` 推断 | 显式 `type` |
| `language_or_subtitle_dependency` | 显式 `Multilingual` domain / sub-category | 不显式，需要从数据内容推断 | 字幕是官方任务设计的一部分 |

## 对 Molmo2 数据选择的含义

三者共享、最值得重点对齐的信号包括：

- 长时长覆盖
- 时序与关系推理
- local-vs-global 范围感知
- 字幕参与或文本交错理解
- 在知识、生活、纪录片、体育、多语言等领域上的平衡覆盖

这份统一画像是后续以下工作的 benchmark 侧基础：

- benchmark 到 Molmo2 的检索召回
- 加权重排
- mixture quota 分配
- 基于短板的 curriculum 构造

## Molmo2 数据画像

### 范围说明

这里的 Molmo2 画像只覆盖和 `VideoMME`、`LVBench`、`LongVideoBench` 最相关的视频子集，不把多图任务混进来。

按和三个 benchmark 的相关性，当前更值得关注的是：

- 第一层：`AskModelAnything`、`Cap`、`VideoCapQA`、`LongCapQA`、`VideoSubtitleQA`
- 第二层：`VideoCountEval`、`VideoPoint`
- 低优先或 eval-only：`CapEval`、`VideoCountEvalExists`、`VideoPointEval`、`VideoTrack`、`VideoTrackEval`

### 主表

| 子集 | 任务形态 | 行数 | 唯一视频数 | 平均行数/视频 | 关键字段 | 备注 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `Molmo2-AskModelAnything` | 开放式视频 QA | `129,470` | `43,210` | `2.996` | `question`, `answer` | 通用 QA 覆盖广，但风格更散 |
| `Molmo2-Cap` | 视频 caption / transcript 聚合 | `107,817` | `102,214` | `1.055` | `merged_caption`, `video_transcript`, `clip_captions`, `frame_captions`, `video_frame_merged_caption` | 不是 QA，但适合长视频摘要和再生成 QA |
| `Molmo2-CapEval` | caption eval | `693` | `692` | `1.001` | `atomic_statements`, `statement_categories`, `aggregated_caption` | eval-only，不建议并入训练池 |
| `Molmo2-VideoCapQA` / `CapQA` | 通用视频 QA | `950,579` | `190,631` | `4.986` | `Question`, `Answer`, `Category`, `NegativeAnswers` | 量大，题型覆盖广 |
| `Molmo2-VideoCapQA` / `LongCapQA` | 长视频 QA | `49,996` | `49,996` | `1.000` | `qa_list` | 每视频一行，平均 `5` 组 QA |
| `Molmo2-VideoSubtitleQA` | 字幕对齐 QA | `468,502` | `102,938` | `4.551` | `Question`, `Answer`, `Category`, `AlignmentType`, `subtitle` | 对 `LongVideoBench` 最关键 |
| `Molmo2-VideoCountEval` | 计数/点标注 QA | `658,873` | `276,299` | `2.385` | `question`, `label`, `count`, `points`, `category`, `video_duration` | 对 `VideoMME` counting 短板有直接帮助 |
| `Molmo2-VideoCountEvalExists` | CountEval 本地视频子集 | `456` | `435` | `1.048` | `question`, `label`, `count`, `clip_start`, `clip_end` | 由 val 过滤出“本地确实有视频”的子集 |
| `Molmo2-VideoPoint` | 点定位 / 引用 /计数 | `659,189` | `276,511` | `2.384` | `question`, `label`, `count`, `points`, `category` | 偏 grounding，不应过采样 |
| `Molmo2-VideoPointEval` | point eval | `181` | `156` | `1.160` | `label`, `points`, `masks`, `count` | eval-only |
| `Molmo2-VideoTrack` | 点轨迹跟踪 | `29,704` | 以轨迹条目为主 | - | `exp`, `points`, `segments`, `fps` | 更像 tracking 训练集 |
| `Molmo2-VideoTrackEval` | 带 mask 的点轨迹 eval | `3,147` | 以轨迹条目为主 | - | `masks`, `points`, `segments` | eval-only |

### Caption 类

#### `Molmo2-Cap`

- 全量行数：`107,817`
- 唯一视频数：`102,214`
- `merged_caption`：`107,817` 行非空
- `video_frame_merged_caption`：`107,817` 行非空
- `video_caption`：`85,454` 行非空
- `video_transcript`：`85,454` 行非空
- `clip_captions`：`107,817` 行非空，非空行平均 `4.08` 段
- `frame_captions`：`107,143` 行非空，非空行平均 `12.836` 帧描述

解释：

- 这类数据不是 benchmark 同型 QA，但很适合补 `LVBench` / `LongVideoBench` 所需的长叙事和摘要能力。
- 如果后续要做 benchmark-style QA 再生成，`Cap` 是最值得先利用的非 QA 子集。

### QA 类

#### `Molmo2-AskModelAnything`

- schema：`video_id`, `question`, `answer`
- 行数：`129,470`
- 唯一视频数：`43,210`

解释：

- 它是比较宽泛的通用视频 QA 池，召回时可以提供覆盖面，但不适合作为长视频能力的唯一主力来源。

#### `Molmo2-VideoCapQA`

`CapQA`：

- 行数：`950,579`
- 唯一视频数：`190,631`
- 原始 `Category` 数：`127`
- 归一化后 `Category` 数：`118`
- Top 类别：
  - `object location`: `52,019`
  - `event location`: `46,964`
  - `human object relationship`: `43,226`
  - `event detection`: `41,238`
  - `scene sequence`: `39,497`
  - `property comparison`: `36,174`
  - `event sequence`: `33,982`
  - `video editing effects`: `32,846`
  - `object reasoning`: `29,700`
  - `human pose`: `27,500`

`LongCapQA`：

- 行数：`49,996`
- 唯一视频数：`49,996`
- `qa_list` 总 QA 对数：`249,978`
- 平均每视频 QA 数：`5.0`

解释：

- `CapQA` 是最像 `VideoMME` 的通用 QA 大池。
- `LongCapQA` 是最像 `LVBench` / `LongVideoBench` 的长上下文 QA 池，应该被单独看待，不要和短视频 QA 混成同一配额。

#### `Molmo2-VideoSubtitleQA`

- 行数：`468,502`
- 唯一视频数：`102,938`
- 原始 `Category` 数：`133`
- 归一化后 `Category` 数：`125`
- Top 类别：
  - `dialogue content`: `32,632`
  - `event causality`: `29,370`
  - `text recognition`: `26,680`
  - `object recognition`: `24,186`
  - `causal reasoning`: `21,488`
  - `object properties`: `19,335`
  - `object interaction`: `18,510`
  - `scene reasoning`: `17,385`
  - `action reasoning`: `16,234`
  - `event detection`: `14,799`

`AlignmentType` 归一化后 Top 类型：

- `temporal sequence bridging`: `98,871`
- `forward alignment (dialogue → visuals)`: `85,200`
- `reverse alignment (visuals → dialogue)`: `83,110`
- `explanation grounding (dialogue → visual effect/state)`: `53,478`
- `cross-modal reasoning (dialogue + visuals → derived answer)`: `42,827`

解释：

- 这是当前 Molmo2 里和 `LongVideoBench` 最直接对齐的子集。
- 它的价值不只是“有字幕”，而是它已经把字幕-视觉对齐拆成了多种 alignment 模式。

### Counting / Grounding 类

#### `Molmo2-VideoCountEval`

- 行数：`658,873`
- 唯一视频数：`276,299`
- `video_duration` 范围：`1.0` 到 `228.5` 秒
- `category` 分布：
  - `object`: `268,175`
  - `action/event`: `175,226`
  - `referring expression`: `60,902`
  - `comparative reference`: `48,015`
  - `indirect reference`: `39,567`
  - `spatial reference`: `27,985`
  - `animal`: `23,799`
  - `anomaly`: `15,204`

#### `Molmo2-VideoCountEvalExists`

- 行数：`456`
- 唯一视频数：`435`
- `video_duration` 范围：`9.5` 到 `188.0` 秒
- `build_report.json` 显示：
  - 原始 val：`533`
  - val 中 YouTube 样本：`484`
  - 实际保留：`456`
  - 保留唯一视频：`435`

#### `Molmo2-VideoPoint`

全量：

- 行数：`659,189`
- 唯一视频数：`276,511`

`train`：

- 行数：`658,340`
- 唯一视频数：`275,837`
- `category` 分布：
  - `object`: `267,749`
  - `action/event`: `175,168`
  - `referring expression`: `60,902`
  - `comparative reference`: `48,015`
  - `indirect reference`: `39,567`
  - `spatial reference`: `27,985`
  - `animal`: `23,750`
  - `anomaly`: `15,204`

`val`：

- 行数：`849`
- 唯一视频数：`724`
- 注意：`val` 的 schema 和 `train` 不同
  - `train` 用 `category`
  - `val` 用 `keyword_category` + `capability`
- `keyword_category`：
  - `object`: `559`
  - `action/event`: `86`
  - `referring expression`: `76`
  - `indirect reference`: `69`
  - `animal`: `59`
- `capability`：
  - `count`: `704`
  - `point`: `145`

按目录配置拆分时，`README.md` 和本地 parquet 一致：

- `action_or_event`: `175,168`
- `animal`: `23,750`
- `anomaly`: `15,204`
- `comparative reference`: `48,015`
- `indirect reference`: `39,567`
- `object`: `267,749`
- `referring expression`: `60,902`
- `spatial reference`: `27,985`

解释：

- `VideoCountEval` 更像“显式计数 / 时间点提示 / 局部证据”。
- `VideoPoint` 更像“点定位 / 引用 / grounding”。
- 两者都不是三个 benchmark 的主干数据，但对修 `VideoMME` 的 counting 和 grounding 弱项很有用。
- 这里有一个工程陷阱：`VideoPoint train/val` schema 不一致，自动化读取时必须分开处理。

### Tracking 类

#### `Molmo2-VideoTrack`

- parquet 文件数：`16`
- 总行数：`29,704`
- 示例 schema：
  - `id`, `video`, `clip`, `video_dataset`, `video_source`
  - `exp`, `obj_id`, `mask_id`, `points`, `segments`
  - `start_frame`, `end_frame`, `w`, `h`, `n_frames`, `fps`

#### `Molmo2-VideoTrackEval`

- parquet 文件数：`5`
- 总行数：`3,147`
- 示例 schema：
  - 与 `VideoTrack` 类似，但额外有 `masks`

解释：

- 这两类更适合“轨迹 / identity persistence / motion grounding”方向。
- 如果目标是 `VideoMME + LVBench + LongVideoBench`，它们不是第一波筛数主力，除非后续 profiling 发现 spatial-grounded continuity 明显是瓶颈。

### 直接结论

如果目标是“最可能提升 `VideoMME`、`LVBench`、`LongVideoBench`”，当前 Molmo2 子集的优先级建议是：

1. `LongCapQA`
2. `VideoSubtitleQA`
3. `CapQA`
4. `Cap`
5. `AskModelAnything`
6. `VideoCountEval`
7. `VideoPoint`
8. `VideoTrack` / 各类 eval 集

更具体地说：

- 面向 `VideoMME`：
  - 主体用 `CapQA`
  - 用 `VideoCountEval` 定向补 counting
  - 用 `AskModelAnything` 提供开放式 QA 覆盖
- 面向 `LVBench`：
  - 主体用 `LongCapQA + Cap`
  - 用 `VideoSubtitleQA` 补长上下文里的文本线索
- 面向 `LongVideoBench`：
  - 主体用 `VideoSubtitleQA + LongCapQA`
  - 需要显式重视 `AlignmentType`、`global/local` 和关系型题目

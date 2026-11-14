# 使用与故障恢复

## 配置书籍

复制 `config/book.example.json` 为 `config/book.json`。`pdf` 和 `ocr` 是相对于配置文件的路径；`pandoc` 可写绝对路径、相对可执行文件路径或 PATH 命令 `pandoc`。每本书使用独立的 OCR 缓存和工作目录。配置、PDF 或页码范围发生变化时使用新的工作目录。

实际章节需在 `sections` 中填入按 PDF 物理页码递增的起始页和标题，例如：

```json
"sections": [
  {"page": 1, "title": "封面与前言", "kind": "front"},
  {"page": 8, "title": "第一章", "kind": "chapter"},
  {"page": 40, "title": "第二章", "kind": "chapter"}
]
```

其他高级字段包括 `running_headers`、`exclude_pages`、`block_roles`、分区阅读方向、`cross_page_joins`、`footnotes` 和 `figures`。框坐标使用按 `ocr_scale` 渲染后的像素；配置页码从 1 开始。跨页连接须显式指定来源块 ID。

导入既有 OCR 时，设 `source_format=legacy`，让 `ocr` 指向页记录 JSONL。步骤 1 此时不加载 OCR 模型。结构 OCR 设 `source_format=structure`，`ocr` 指向独立缓存目录；默认 GPU 设备为 `gpu:0`，也可显式 `--device cpu`。

## 接入两个模型

`.env.local` 是按行读取的数据文件，不会被当作脚本执行：

```dotenv
AI_BASE_URL=https://your-primary-provider.example/v1
AI_MODEL=your-primary-vision-model
AI_API_KEY=fill-locally
JUDGE_BASE_URL=https://your-judge-provider.example/v1
JUDGE_MODEL=your-independent-vision-model
JUDGE_API_KEY=fill-locally
```

同名环境变量优先于文件。必须使用 HTTPS，禁止在地址里嵌入用户名、密码或 query。服务需支持 `/chat/completions`、JSON 输出和 OpenAI-compatible 的文本/图片消息。具体模型支持范围与计费以服务方为准；并非任意标称兼容的中转都支持所有请求参数。

模型只通过 `--submit` 调用。服务端可能忽略低思考/禁用思考请求，实际消耗记录在返回的 `usage` 中。图文阶段发送裁片和对应文字；文本阶段发送分段文字或疑点及上下文。不要对没有上传授权的文档启用外部模型。

若第 5 步另选文本模型，定义 `TEXT_JUDGE_BASE_URL`、`TEXT_JUDGE_MODEL`、`TEXT_JUDGE_API_KEY`，并给第 5 步添加 `--judge-prefix TEXT_JUDGE`。角色前缀也可以用于其他 OpenAI-compatible 服务，不必是 DeepSeek。

## 续跑与导入

重复同一个阶段命令会复用来源相同的结果。模型、提示词、有效输入或上游意见改变时会创建新版本目录。原始模型响应即时保存；收到但无法解析的响应不会自动重复计费请求。

临时网络失败可在同一 `workflow` 命令中加 `--retry-failed`。该开关只重试尚未收到响应的临时网络/429/500/502/503/504 错误，不重试身份错误或已经收到的截断回答。网络失败也可能已在服务端计费，脚本无法替服务方保证失败请求免费。

有缺失/无效复核结果时，默认不应用。若接受先生成阅读稿，给第 3 或 5 步加 `--allow-pending --apply`，这些位置保留原文并在台账标明。有效的 `uncertain` 结论始终保留原文，不会猜补。不要直接修改 `corrected/` 中的快照或补写校验和绕过审核表。

每阶段的实际路径和结果在 `workflow.json` 中。长任务执行期间可以查看对应版本目录的 `results/` 和 `summary.json`，不要同时启动同一工作目录的两条 workflow 命令。

## 停止文本或独立复核

使用当前阶段版本目录，不是最外层工作目录：

```bash
bookocr text-review stop --out output/my-book/04-text/<version>
```

当前小批次已经发出的请求会先保存，随后停止继续提交。独立复核阶段可在相应的 `03-judge/<version>` 或 `05-judge/<version>` 中创建内容任意的 `STOP` 文件，同样在当前批次结束后停止。恢复时只删除该标记，再运行原来的命令。

不要将强行结束进程当作日常停止方式，强杀会失去尚未返回的请求结果。已落盘的原始响应仍可复用。

## 独立于六步入口使用

保留原有 `recognize`、`layout`、`quality`、`structure`、`simplify`、`epub`、`audit` 和 `recheck` 命令，用于单阶段排查；`text-review prepare|submit|report|stop` 也可以独立使用。独立文本命令默认采用历史 `DEEPSEEK_*` 设置，六步入口采用 `AI_*` 与 `JUDGE_*`。

外部模型/agent 也可读取第 4 步的 `evaluation_tasks.jsonl`，向 `evaluation_template.csv` 写入 `agree/disagree/uncertain`。重新导出会保留既有评测值。内置第 5 步以严格验证的原始模型回答为依据，不会把手工改过的模板或 `completed` 标签直接当成可应用的批准。若需人工勘正，可使用第 3 步的正式审核表并正常重建；不要直接编辑已缓存的模型判定。

## 常见问题

- **没有置信度**：某些结构块没有直接分数，不能据此当作高置信度通过；查看页级报告和原始 OCR 分配情况。
- **引文无法唯一定位**：原始响应保留，检查模型是否改写了引用；不要用模糊匹配自动改书。
- **结果以 length 结束**：输出可能被思考预算耗尽或内容太长；保留回答，调整模型/分段后在新目录重跑，不将半截输出当成零问题。
- **CPU/GPU 导入失败**：基础后处理不依赖 Paddle；OCR 需另装兼容后端。可先用已有 OCR 或离线演示验证后半条流水线。
- **EPUB 图像丢失**：不要只移动 Markdown 而遗漏同目录 `images/`；检查最终 `audit_report.json`。

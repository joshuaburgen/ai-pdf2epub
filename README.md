# AI PDF → EPUB

面向中文扫描书籍的 **AI 辅助 PDF 转简体 EPUB** 工具包。OCR、模型审查、独立复核、勘正导入和出版分成六步，每步保留原始数据与来源记录，可以单独重跑。

目标是生成可顺畅阅读的电子书。最后的文本审查只补前面步骤的明显遗漏，不润色、不改译法、不统一术语，也不让第二模型扩大修改范围。

## 六步流程

```mermaid
flowchart LR
  A[1 OCR与版面恢复] --> B[2 主模型看图审疑点]
  B --> C[3 独立视觉复核并导入]
  C --> D[繁转简]
  D --> E[4 全文保守补漏]
  E --> F[5 复核与最小修补]
  F --> G[6 EPUB打包与结构检查]
```

| 步骤 | 做什么 | 使用什么 |
| --- | --- | --- |
| 1 | OCR、置信度、文字框、阅读顺序、图片及疑点清单 | PP-StructureV3，可用 GPU；也能导入已有 OCR |
| 2 | 对照原图审查低置信度及规则命中的疑点块 | 支持图像输入的主模型，例如 DeepSeek |
| 3 | 字形/全半角等价变化可直接导入；其余交独立模型看图复核，再通过正式审核表应用 | 独立视觉模型；不把模型自报置信度当准确率 |
| 4 | 分段覆盖完整简体文本，只查明显错字、漏字、重复、机械顺序和严重标点错误 | 主模型，仅提出疑点 |
| 5 | 复核疑点，只应用必要的最小局部修补 | 独立模型，或主模型参照原图复核；不增添编辑性修改 |
| 6 | 从最终勘正文本生成封面、目录、配图与可重排 EPUB，检查链接和包结构 | Pandoc、OpenCC、PyMuPDF |

默认步骤 2–5 支持 OpenAI-compatible Chat Completions；第五步可选的原图补审使用 Responses 图像接口。主模型和独立复核模型可来自不同服务；独立路线要求服务/模型标识不同，但无法替用户判断哪个模型更强。步骤 3 的模型必须支持 `image_url`，步骤 5 可以另选文本模型。

## 安装

需要 Python 3.11+ 和 [Pandoc](https://pandoc.org/installing.html)。基础安装不包含 GPU 模型：

```bash
git clone https://github.com/joshuaburgen/ai-pdf2epub.git
cd ai-pdf2epub
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e .
```

需要实际 OCR 时再安装：

```bash
python -m pip install -e '.[structure]'
```

PaddlePaddle CPU/GPU 构建需按照 [PaddleOCR 安装指引](https://github.com/PaddlePaddle/PaddleOCR) 和本机 CUDA 环境单独选择。繁转简与 EPUB 打包使用 CPU。可选第二 OCR 引擎为 `.[second-engine]`，它只提供对照，不是最终裁决。

## 先跑一个不收费的完整演示

```bash
bookocr demo --out output/demo
```

演示现场生成原创的两页 PDF、几何配图与模拟 OCR 数据，模拟两个模型的回答，真实执行两轮勘正导入、OpenCC 和 Pandoc 出版。它不会调用 API、不会下载 OCR 模型，也不代表真实模型的质量评测。控制台会输出 EPUB 和最终 Markdown 路径。

Pandoc 不在 PATH 时，可指定 `bookocr demo --out output/demo --pandoc /path/to/pandoc`。

## 转换自己的书

1. 将自己有权处理的 PDF 放入 `input/`。
2. 复制 `config/book.example.json` 为 `config/book.json`，填写书名、PDF 路径、OCR 缓存目录和章节配置。配置内路径相对于配置文件，`pandoc` 也可以直接使用 PATH 命令。
3. 复制 `.env.example` 为 `.env.local`，填写主模型 `AI_*` 和复核模型 `JUDGE_*`。该文件被 Git 忽略；不要把密钥放在命令参数中。

按顺序执行：

```bash
# 1. 真实 OCR；已有符合配置的缓存会复用。
bookocr workflow --config config/book.json --out output/my-book --stage 1 --device gpu:0

# 2. 主模型视觉复核。
bookocr workflow --config config/book.json --out output/my-book --stage 2 --submit

# 3. 独立视觉模型复核，并正式导入勘正、重建繁简 Markdown。
bookocr workflow --config config/book.json --out output/my-book --stage 3 --submit --apply

# 4. 全文保守补漏，仅标疑点。
bookocr workflow --config config/book.json --out output/my-book --stage 4 --submit

# 5. 独立评测疑点，仅导入通过的最小修补。
bookocr workflow --config config/book.json --out output/my-book --stage 5 --submit --apply

# 6. 使用第五步的最终文本打包，不会重新生成文本覆盖补漏结果。
bookocr workflow --config config/book.json --out output/my-book --stage 6
```

**`--submit` 才会发出收费 API 请求；`--apply` 才会在步骤 3、5 导入勘正。** 去掉开关可以先准备、检查任务。默认处理全书；建议先用另一个输出目录运行第 1 步 `--pages 1-10` 验证版面，再新建全书输出目录。后续阶段保持该次选页范围，样本 EPUB 会标记为部分书籍。

没有可靠返回或意见相互冲突时，默认保留原文。缺失/无效响应会阻止默认导入；若接受先出阅读稿，可显式加 `--allow-pending` 保留这些原文继续。`uncertain` 本身是有效复核结论，表示不改。断点续跑和故障处理见 [使用指南](docs/USAGE.md)。

最终文件位置记录在 `output/my-book/workflow.json` 的 `artifacts.epub` 和 `artifacts.final_markdown`。各阶段使用独立目录；输入或模型切换后会建立新的审查版本，保留旧结果。

## 第四步的边界

第五步也可选择让主模型参照原图复核所有候选：`bookocr workflow --config config/book.json --out output/my-book --stage 5 --source-images --submit --apply`。这不是独立模型复审。程序只自动应用模型明确看清原图、确认必要且通过原文定位和最小范围校验的修补；其余原文保留，任务写入 `next_stage.jsonl`。单独处理已有审查清单见[原图补审指南](docs/SOURCE_REVIEW.md)。

- 只查前三步遗漏、明确妨碍理解的机械性错误；整段零问题是正常结果。
- 能读通但拗口的译文不改，不做更自然的表达替换。
- 不改译法、术语、人物年代、理论、逻辑或作者观点。
- 确定的修补通常各限 16 字；删除重复文本的原片段最多 40 字。没有依据时留空并标记查图，不猜补文字。
- 第五步即使认为建议更通顺，也必须拒绝超范围改稿。重叠且相互冲突的修补保留原文。

## 文档与验证

- [六步流程与数据契约](docs/PIPELINE.md)
- [运行、续跑、停止及配置](docs/USAGE.md)
- [安全与数据范围](SECURITY.md)
- [开发与测试](CONTRIBUTING.md)
- [实现来源与依赖](NOTICE.md)

测试包含错误定位、输入签名、断点复用、独立复核、最小修补、冲突保留及真实 Pandoc 的六步端到端演示。CI 不调用真实模型、不使用商业书籍、不下载 GPU 模型；其通过不能证明任意书籍的 OCR 准确率或任意模型的纠错效果。真实 API 流程需要用户自行配置模型。

章节、页眉、图片、脚注等需要按书籍配置；本项目不声称对任意复杂版面自动恢复无误。OCR 分数不是准确率，分段覆盖全文也不等于查明了所有跨章矛盾。EPUB 包检查通过代表结构可用，不代表逐字校准完成。

## 许可证

[AGPL-3.0-or-later](LICENSE)。项目不分发输入书籍、扫描图片、OCR 权重、Pandoc 二进制或 API 密钥；依赖和权重保留各自许可证。

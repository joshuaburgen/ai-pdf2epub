# 候选疑点参照原图复核

这是六步流程中第五步的可选路线：全部候选进入原图核对，确定的最小勘正自动应用，不确定的留到下一步。可以使用提出疑点的同一模型，不属于独立评测；原来的独立模型路线仍可使用。

## 在六步流程中使用

```powershell
bookocr workflow --config config/book.json --out output/my-book --stage 5 --source-images --submit --apply
bookocr workflow --config config/book.json --out output/my-book --stage 6
```

使用 `AI_*` 配置主模型。模型需支持 Responses 图像输入；请求明确设置 `reasoning.effort=none`。`--submit` 发送 API 请求，`--apply` 才生成修补后的阅读稿。缺失或无效返回默认阻止流程应用；明确接受保留这些原文时可加 `--allow-pending`。有效的“不确定”判断会自然进入后续队列。

## 对已有全文审查结果单独使用

```powershell
bookocr source-review prepare --from-review output/text-review --pdf input/book.pdf --out output/source-review --env-file .env.local
bookocr source-review submit --out output/source-review --env-file .env.local --workers 3
bookocr source-review report --out output/source-review
bookocr source-review apply --out output/source-review --target output/source-corrected
```

单独命令默认使用 `DEEPSEEK_*`，可用 `--api-prefix AI` 切换。`submit --limit 3` 先验证清单前 3 项，随后去掉限制续跑，已保存的收费响应会复用。`--retry-failed` 仅重试无响应的暂时网络错误；格式不合规、截断等已付费返回不会静默重发。`stop --out ...` 让当前批次保存后停止；删除该目录的 `STOP` 文件再续跑。

## 自动应用条件

- 引用精确匹配冻结正文，页码来自正文的 PDF 页码标记；页码是物理页码，须与输入 PDF 一致。
- 每页提供全页预览及重叠高清上下分片；漏字、顺序、断句问题额外提供相邻页。PDF、正文、候选、图片、提示词、接口参数和模型均绑定缓存身份。
- 模型必须返回 `correct`、`certainty=certain`、`source_readable=true`，并给出实际印刷文字、页码和图像依据。印刷文字经 OpenCC 转简后须与替换一致。这些是模型的判断，不能当作准确率保证。
- 每项只允许一个可唯一定位的最小替换，通常两侧各最多 16 字；重复删除原片段最多 40 字。禁止跨行改写、改 Markdown 结构和链接，禁止润色、改译法或凭语义补写。
- 相同补丁只应用一次；不同补丁重叠时均保留原文。模型确定原文正确时结束该项；看不清、无法定位、格式不合规、修改冲突或其他不确定情况进入后续队列。

`apply` 创建独立阅读稿，保留输入 Markdown、PDF、OCR 和旧 EPUB。重新运行已应用结果会校验指纹并复用；新增模型结果后须选择新的目标目录。单独命令允许未完成项保留原文；查看返回的 `missing_or_invalid` 和队列数量。

## 输出

- 审查目录：`tasks.json`、`images/`、逐项 `results/`、`summary.json`、`report.md`、`decisions.csv`、`next_stage.jsonl`。
- 应用目录：`book_simplified.md`、原有正文 `images/`、`source_patch_ledger.json`、`summary.json`、`next_stage.jsonl`。
- 应用后的队列包括冲突项，优先使用应用目录的队列交给下一模型。原字符坐标仍指向冻结正文，下一阶段应从新稿重新定位，不能直接沿用旧偏移。

本阶段只审查已有疑点，不会再次通读全书查找新问题，也不会自动调用下一模型。接口和可复用实现调研见 [RESEARCH_VISUAL_REPAIR.md](RESEARCH_VISUAL_REPAIR.md)。

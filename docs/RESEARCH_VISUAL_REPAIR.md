# Optional source-image verification of residual OCR findings

Verified 2026-11-15 against primary sources. This note contains no book content, provider credentials, or live inference results.

## DeepSeek contract

- Use `POST https://api.deepseek.com/responses`, with `model: "deepseek-flash"`. A user message can contain `{"type":"input_text","text":"..."}` and `{"type":"input_image","image_url":"data:image/png;base64,...","detail":"high"}`. Images in assistant messages are rejected; `image_url` and `file_id` are mutually exclusive. The service is stateless; keep checkpoints locally. [Responses guide](https://api-docs.deepseek.com/guides/responses_api/)
- Set `reasoning: {"effort":"none"}` explicitly: `low` enables thinking. Request JSON with `text: {"format":{"type":"json_schema","name":"residual_visual_verdict","schema":{...}}}`; `json_object` is also documented. The reference does not document a `strict` field here. Check completion status and validate parsed results locally before allowing any patch. [Responses reference](https://api-docs.deepseek.com/api/create-response/)
- `deepseek-v4-flash-vision-exp` is a retired alias routed to the latest Flash model. PNG/JPEG/GIF/WebP are supported. `high` keeps the original image at the detail-selection stage, but inference still resizes large images to roughly 1300×1300 total pixels. Prefer legible focused source crops with neighboring text over a dense whole page. Inline base64 counts toward the 48 MiB body limit; inline images are limited to 32 MiB each. [Vision guide](https://api-docs.deepseek.com/guides/vision/)

These are documentation findings, not evidence that any configured third-party gateway implements the same contract. Test the selected endpoint/model before a batch; fail closed on unsupported images or formats. JSON schema conformity proves shape, not transcription accuracy.

## Existing implementations considered

- [yuhanwang14/PDF-to-EPUB](https://github.com/yuhanwang14/PDF-to-EPUB) implements PaddleOCR PP-StructureV3 → book-specific structural configuration → Pandoc EPUB, with footnotes, navigation, embedded plates and resumable OCR. Its [rebuild script](https://github.com/yuhanwang14/PDF-to-EPUB/blob/main/skills/PDF-to-EPUB/scripts/rebuild_md_from_json.py) reconstructs per-page Markdown from saved JSON, preferring ordered parsing blocks. Reuse the checkpoint/configuration pattern; this does not provide the desired source-anchored residual-findings verification contract.
- [huutrungle2001/ChineseBookOCR](https://github.com/huutrungle2001/ChineseBookOCR) is the repository already named in this project's NOTICE. Its README documents Traditional Chinese vertical RTL column processing, PaddleOCR/CnOCR discrepancy detection and image-backed review prompt packages. [APPROACH.md](https://github.com/huutrungle2001/ChineseBookOCR/blob/main/APPROACH.md) emphasizes preserving raw OCR and retaining uncertain lines for human review. Reuse the focused-crop/evidence workflow concept; its described copy-paste ChatGPT/Gemini review workflow is not a ready-made DeepSeek Responses client or transactional patcher.

Neither reviewed repository establishes the safety of automatically applying a model's claimed confidence. Existing `bookocr.text_review`, source maps, crop rendering, storage signatures and final patch ledger are closer integration points than importing a second OCR pipeline. See [pipeline contract](PIPELINE.md), [text review](../bookocr/text_review.py), [crop helper](../bookocr/review.py), and [workflow](../bookocr/workflow.py).

## Recommended integration boundaries

Design recommendation, rather than a provider guarantee:

1. Keep this stage optional. Consume only already anchored residual OCR/mechanical findings; preserve the narrow scope and minimal-edit limits. Never ask the vision model to rewrite surrounding prose or adjudicate facts, style, translations or authorial wording.
2. Freeze source Markdown, findings, PDF identity, source mapping, crop settings, model identity and prompt/schema version in a manifest. Resolve the issue to mapped PDF pages/blocks; ambiguous or missing mappings become unresolved queue entries. Do not substitute a page inferred only from textual meaning.
3. Send source-PDF image crops plus the exact existing excerpt and candidate. Ask for visible source transcription, `confirm/reject/uncertain`, and explicit image evidence. A simplified final manuscript must compare the image's Traditional transcription through the configured OpenCC conversion; visual Traditional/Simplified differences alone are not OCR errors.
4. Autoapply only a confirmed, locally valid minimal candidate with a legible source reading and a unique unchanged anchor. Require exact candidate agreement, frozen-source hash, bounded edits, non-overlap, preserved Markdown structure and an auditable ledger. A model's confidence number alone is insufficient.
5. Queue rejected, uncertain, illegible, conflicting, unanchored, truncated, malformed and failed jobs with reason, page/crop reference, raw response and candidate. Keep original text. Deduplicate and resume completed jobs by manifest/request signatures; do not silently change model or protocol while resuming.
6. Apply to a new final Markdown artifact through the existing patch machinery, then rebuild EPUB from that artifact. Preserve raw OCR, earlier Markdown and the original findings. Record zero accepted changes as a valid outcome rather than forcing a correction.

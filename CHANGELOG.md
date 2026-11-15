# Changelog

## 0.3.0

- Add source-image verification of every residual OCR finding, using Responses API image input with reasoning disabled.
- Automatically apply only certain, image-supported minimal corrections; preserve originals and export evidence-backed follow-up tasks.
- Add `workflow --stage 5 --source-images` and standalone `source-review` commands.
- Add explicit unresolved-only image review refinement, preserving closed paid responses and versioned evidence.
- Add structured retry judgments, neighboring pages, high-resolution quadrant images, traditional-to-simplified anchors and atomic multi-patch application.
- Validate source identities, source readings, patch bounds, duplicate patches and conflicts; cover the new route with offline real-Pandoc integration tests.

- Refine only incomplete text-review ranges into smaller chunks, reuse valid paid responses, preserve existing findings and cumulative usage, and validate complete source coverage.
- Allow standalone text-review commands and the batch runner to select an API environment prefix.

## 0.2.0

- Publish a clean open-source package without book data or private history.
- Add a six-stage `bookocr workflow` CLI with source-bound versioned artifacts.
- Add independent visual judgment and formal correction-ledger imports.
- Add conservative full-text residual OCR checks and independent text judgment.
- Apply only approved minimal text patches; retain uncertain, missing, and conflicting repairs.
- Publish final corrected text without regenerating over stage-five patches.
- Add an original offline six-stage demo with real Pandoc export and no API/GPU calls.
- Add Windows/Linux-oriented configuration, independent provider prefixes, CI, packaging and licensing.

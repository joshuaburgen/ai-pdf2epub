# v0.3.0 — source-image verification and targeted re-review

Every residual OCR finding can now be checked against its source PDF images. Clear, image-supported minimal repairs are applied to a new manuscript, while unresolved findings retain their original text and carry source evidence into the next review stage.

- New `bookocr source-review` prepare/submit/report/apply/stop commands and `workflow --stage 5 --source-images` route.
- Responses API image requests with reasoning disabled, cached raw responses and source-bound task identities.
- `source-review refine` retries only unresolved findings with structured JSON, adjacent PDF pages and high-resolution quadrant images. Closed paid responses remain cached.
- Traditional/simplified source anchors, bounded multi-edit repairs, duplicate handling and atomic conflict retention.
- Original PDF, OCR and earlier manuscripts are preserved; correction ledgers and follow-up tasks carry image references.
- Also includes smaller-chunk retries for incomplete full-text reviews and configurable API environment prefixes.

Validation: 95 local tests, including real Pandoc integration using original offline fixtures. Release packages contain generic code and documentation, not books, scans, real API responses or credentials. Model certainty is a judgment rather than an accuracy guarantee; source inconsistencies still block automatic repair.

See [SOURCE_REVIEW.md](SOURCE_REVIEW.md) for commands and the review contract. Existing independent-model review remains available.

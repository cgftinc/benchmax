# examples

- [math](math/) — a sanity-check environment that extends `BaseEnv` to demonstrate datasets, tool use, deterministic rewards, local validation, and hosted validation.
- [dominant-color](dominant-color/) — a multi-turn vision environment that returns images from tool calls and scores ordered visual memory.
- [geo3k](geo3k/) — a multimodal geometry environment that resolves a public hugging face dataset at runtime and exposes an image-crop tool.
- [qwen3-ocr-reverse](qwen3-ocr-reverse/) — a single-turn vision environment with uploaded synthetic pages and deterministic OCR rewards.
- [telestich](telestich/) — a group-scored poetry environment with deterministic checks, an llm judge, and a feedback tool.
- [postgres-search](postgres-search/) — a `BaseEnv` library for retrieval training against a provisioned corpora-service corpus.
- [gitlab_handbook_bm25](gitlab_handbook_bm25/) — a pinned scheduler A/B for the GitLab handbook BM25 environment with and without the TITO Gateway.
- [aime](aime/) — a `HarborEnv` where an offline-installed mini-swe agent solves AIME math inside modal sandboxes.
- [harvey](harvey/) — a `HarborEnv` training on harvey's LAB legal tasks with its native harness loop and rubric judge.

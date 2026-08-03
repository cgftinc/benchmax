# examples

- [math](math/) — a sanity-check environment that extends `BaseEnv` to demonstrate datasets, tool use, deterministic rewards, local validation, and hosted validation.
- [dominant-color](dominant-color/) — a multi-turn vision environment that returns images from tool calls and scores ordered visual memory.
- [geo3k](geo3k/) — a multimodal geometry environment that resolves a public hugging face dataset at runtime and exposes an image-crop tool.
- [qwen3-ocr-reverse](qwen3-ocr-reverse/) — a single-turn vision environment with uploaded synthetic pages and deterministic OCR rewards.
- [telestich](telestich/) — a group-scored poetry environment with deterministic checks, an llm judge, and a feedback tool.
- [neon_rag](neon_rag/) — an end-to-end retrieval environment with versioned Neon Lakebase ingestion, grounded QA generation, hybrid search, and citation rewards.
- [turbopuffer_rag](turbopuffer_rag/) — the same retrieval-training pattern with example-local lexical, vector, and hybrid TurboPuffer search.
- [chroma_rag](chroma_rag/) — a current dense-vector Chroma Cloud/self-hosted example with explicit embeddings and static capabilities.
- [pinecone_rag](pinecone_rag/) — a vector-only Pinecone example that targets the current data-plane host API directly.
- [aime](aime/) — a `HarborEnv` where mini-swe agent solves AIME math problem.
- [harvey](harvey/) — a `HarborEnv`where harvey's native harness is used to solve harvey's LAB legal tasks.

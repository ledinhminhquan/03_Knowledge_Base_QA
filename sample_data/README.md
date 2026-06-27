# `sample_data/` — committed synthetic samples

Small, fully synthetic files (no private data) to try the system without any
download:

```bash
# Ingest the sample docs then ask a question (the agent seeds the built-in KB too)
kbqa ask --question "Who designed the Eiffel Tower?" --config configs/infer.yaml

# Run the agent over the built-in sample QA (some answerable, one not)
kbqa demo-agent --config configs/infer.yaml
```

`sample_docs.jsonl` holds three short factual documents; `questions.txt` includes
four answerable questions and one (`capital of Australia`) that is **not** in the
KB — the agent should abstain ("I don't know") on that one rather than guess.

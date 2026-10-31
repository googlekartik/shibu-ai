# Cloud LoRA / QLoRA workflow (prepared, not started)

This directory is separate from the Flask runtime. It does **not** run during a
Render build or app startup, and it contains no user conversations or uploaded
documents. No authorised training dataset was supplied with this project, so no
training job has been launched.

## Required approval gate

Before starting a hosted GPU job, the owner must provide an approved dataset
path or Hugging Face dataset revision, confirm the licence permits training and
hosting, name the cloud/GPU budget, and approve the base model. Do not use chat
history or RAG uploads as a substitute dataset.

Run this non-training validation step in any normal Python environment:

```bash
python training/validate_dataset.py --input /path/to/approved.jsonl
```

It checks JSONL structure, duplicates, empty examples, basic secret-like
patterns, and makes a deterministic train/validation split. It writes only a
redacted quality report unless `--write-cleaned` is explicitly supplied.

## Hosted run proposal

The starting point is `configs/cloud_lora_sft.yaml`:

- Base: `Qwen/Qwen2.5-1.5B-Instruct` (change only after licence review).
- Method: supervised fine-tuning with PEFT LoRA; 4-bit QLoRA is enabled only
  when the selected hosted GPU runner supports it.
- Pilot: 100–500 approved examples, one epoch, bounded sequence length.
- Persistence: write adapter checkpoints and final adapter weights to the
  managed job output/artifact store—not the Render filesystem.
- Promotion: compare the base and adapter using a held-out evaluation set;
  keep OpenAI/HF hosted inference as the production fallback unless the owner
  explicitly promotes the adapter after evaluation.

Use a managed Hugging Face-compatible GPU environment (for example a paid
Space/Job or another approved managed runner) rather than Render or a personal
computer. The hosted service must report GPU type, VRAM, elapsed time and cost
before the run. Fine-tuning is deliberately not included in `requirements.txt`.

## Rollback

Production never replaces `OPENAI_MODEL` or `HF_MODEL` automatically. If an
adapter is promoted later, retain the prior model identifier and set an explicit
`HF_MODEL`/provider value to roll back. Keep the evaluation report and adapter
revision beside the cloud job record.

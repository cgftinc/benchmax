"""Text-only LoRA SFT entry point for the Qwen route-scoring contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def train_qwen_router(
    *,
    train_file: Path,
    eval_file: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    max_sequence_length: int = 8192,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> dict[str, Any]:
    """Fine-tune a text-only LoRA adapter with assistant-only loss."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ValueError(
            "training dependencies are missing; run `uv sync --extra training`"
        ) from error

    train_rows = _read_examples(train_file)
    eval_rows = _read_examples(eval_file)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    except ValueError:
        # Some Qwen releases register the same text-capable checkpoint through
        # the multimodal auto class. The forward path below remains text-only.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    class RouterDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            messages = self.rows[index]["messages"]
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            input_ids = input_ids[:max_sequence_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        longest = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = longest - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [tokenizer.pad_token_id] * padding
            )
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(
        use_cuda
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RouterDataset(train_rows),
        eval_dataset=RouterDataset(eval_rows),
        data_collator=collate,
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "model_name": model_name,
        "method": "supervised_fine_tuning",
        "adapter": "lora",
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_metrics": train_result.metrics,
        "output_dir": str(output_dir),
    }
    (output_dir / "castform-training-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _read_examples(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing SFT dataset: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        messages = row.get("messages") if isinstance(row, dict) else None
        if (
            not isinstance(messages, list)
            or len(messages) < 3
            or messages[-1].get("role") != "assistant"
        ):
            raise ValueError(
                f"{path}:{line_number} must contain system, user, and assistant messages"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"SFT dataset is empty: {path}")
    return rows

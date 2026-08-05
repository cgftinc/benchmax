"""Small LoRA fine-tune for the Qwen scorer."""

from __future__ import annotations

import json
from pathlib import Path


def train_qwen(*, dataset: Path, model: str, output: Path, epochs: float = 1.0) -> None:
    """Fine-tune a causal Qwen checkpoint with LoRA on chat-format examples."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ModuleNotFoundError as error:
        raise RuntimeError("install with: uv sync --extra training") from error

    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=8192,
        padding=True,
        return_tensors="pt",
    )
    prompt_lengths = [
        len(
            tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
            )["input_ids"]
        )
        for prompt in prompts
    ]
    sequence_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if any(
        prompt_length >= sequence_length
        for prompt_length, sequence_length in zip(prompt_lengths, sequence_lengths, strict=True)
    ):
        raise ValueError("a training example was truncated before its assistant target")

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            input_ids = encoded["input_ids"][index]
            attention_mask = encoded["attention_mask"][index]
            labels = input_ids.clone()
            labels[: prompt_lengths[index]] = -100
            labels[attention_mask == 0] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    base = AutoModelForCausalLM.from_pretrained(model)
    tuned = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainer = Trainer(
        model=tuned,
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
        ),
        train_dataset=Dataset(),
    )
    trainer.train()
    tuned.save_pretrained(output)
    tokenizer.save_pretrained(output)

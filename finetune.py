"""
finetune.py — QLoRA fine-tuning of a causal language model on an RTX 5060.

Usage
-----
  # Use defaults from config.yaml:
  python finetune.py

  # Override individual settings:
  python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                     --dataset yahma/alpaca-cleaned \
                     --epochs 1 \
                     --output ./my-adapter

Architecture
------------
  * 4-bit NF4 quantisation via bitsandbytes (QLoRA)
  * LoRA adapters injected with PEFT
  * Supervised fine-tuning via TRL's SFTTrainer
  * bf16 mixed-precision (RTX 5060 Blackwell supports bf16 natively)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides on top of the YAML config."""
    overrides = {
        "model.name": args.model,
        "dataset.name": args.dataset,
        "training.output_dir": args.output,
        "training.num_train_epochs": args.epochs,
        "training.per_device_train_batch_size": args.batch_size,
        "training.learning_rate": args.lr,
        "lora.r": args.lora_r,
    }
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        keys = dotted_key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg


def check_gpu() -> torch.device:
    """Verify that a CUDA-capable GPU is available and log its details."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected.  Make sure the NVIDIA drivers and CUDA toolkit "
            "are installed and that the container was started with --gpus all."
        )
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    logger.info(
        "GPU: %s  |  Compute capability: %d.%d  |  VRAM: %.1f GiB",
        props.name,
        props.major,
        props.minor,
        props.total_memory / 2**30,
    )
    return device


def format_alpaca_prompt(example: dict) -> dict:
    """
    Convert an Alpaca-style record into a single text field.

    Expected keys: instruction, input (optional), output.
    """
    instruction = example.get("instruction", "").strip()
    context = example.get("input", "").strip()
    response = example.get("output", "").strip()

    if context:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
    return {"text": prompt}


def load_and_prepare_dataset(cfg: dict) -> DatasetDict:
    """Download (or load from disk) and prepare the dataset."""
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    text_col = ds_cfg.get("text_column")
    sample_frac = float(ds_cfg.get("sample_fraction", 1.0))
    val_split = float(ds_cfg.get("val_split", 0.05))

    logger.info("Loading dataset: %s", name)

    # Support local files (JSON / JSONL / CSV)
    if Path(name).exists():
        suffix = Path(name).suffix.lower()
        fmt = "json" if suffix in {".json", ".jsonl"} else "csv"
        ds = load_dataset(fmt, data_files=name)
    else:
        ds = load_dataset(name)

    # Flatten to a single split if necessary
    if isinstance(ds, DatasetDict) and "train" not in ds:
        split_name = next(iter(ds))
        ds = DatasetDict({"train": ds[split_name]})

    train_ds: Dataset = ds["train"]

    # Optional sub-sampling
    if 0.0 < sample_frac < 1.0:
        n = max(1, int(len(train_ds) * sample_frac))
        train_ds = train_ds.shuffle(seed=42).select(range(n))
        logger.info("Sub-sampled to %d rows (%.0f%%)", n, sample_frac * 100)

    # Format to a single "text" column when no explicit column is specified
    if text_col is None:
        # Detect Alpaca-style datasets
        if "instruction" in train_ds.column_names:
            logger.info("Applying Alpaca prompt formatter.")
            train_ds = train_ds.map(format_alpaca_prompt, remove_columns=train_ds.column_names)
        else:
            raise ValueError(
                "Dataset has no 'instruction' column and 'text_column' is not set "
                "in config.yaml.  Please set dataset.text_column to the column that "
                "contains the training text."
            )
    else:
        # Rename to "text" if necessary
        if text_col != "text":
            train_ds = train_ds.rename_column(text_col, "text")

    # Create validation split
    if "validation" in ds:
        val_ds = ds["validation"]
        if text_col is None and "instruction" in val_ds.column_names:
            val_ds = val_ds.map(format_alpaca_prompt, remove_columns=val_ds.column_names)
        elif text_col and text_col != "text":
            val_ds = val_ds.rename_column(text_col, "text")
    else:
        split = train_ds.train_test_split(test_size=val_split, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]

    logger.info("Train rows: %d  |  Val rows: %d", len(train_ds), len(val_ds))
    return DatasetDict({"train": train_ds, "validation": val_ds})


def build_bnb_config(cfg: dict) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig for 4-bit QLoRA, or return None for full precision."""
    model_cfg = cfg["model"]
    if not model_cfg.get("load_in_4bit", True):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer(cfg: dict, bnb_config: Optional[BitsAndBytesConfig]):
    """Load the base model and tokenizer from the Hub."""
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure there is a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token (%s)", tokenizer.eos_token)

    logger.info("Loading model: %s", model_name)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Required before adding LoRA to a quantized model
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
        )

    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    """Construct the PEFT LoRA configuration."""
    lora_cfg = cfg["lora"]
    target_modules = lora_cfg.get("target_modules", "all-linear")
    # "all-linear" is a special TRL/PEFT string; pass it through as-is
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def build_training_args(cfg: dict) -> SFTConfig:
    """Build TRL's SFTConfig (a TrainingArguments subclass) from the config."""
    t = cfg["training"]
    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        report_to=t.get("report_to", "none"),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        group_by_length=t.get("group_by_length", True),
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=cfg["model"].get("max_seq_length", 2048),
        packing=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of a causal LLM on an RTX 5060."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model", default=None, help="Override model.name in config.")
    parser.add_argument("--dataset", default=None, help="Override dataset.name in config.")
    parser.add_argument("--output", default=None, help="Override training.output_dir in config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help="Override per_device_train_batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate.")
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=None,
                        help="Override LoRA rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    # ------------------------------------------------------------------
    # 2. GPU check
    # ------------------------------------------------------------------
    check_gpu()

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    datasets = load_and_prepare_dataset(cfg)

    # ------------------------------------------------------------------
    # 4. Model + tokenizer
    # ------------------------------------------------------------------
    bnb_config = build_bnb_config(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg, bnb_config)

    # ------------------------------------------------------------------
    # 5. LoRA
    # ------------------------------------------------------------------
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    training_args = build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )

    logger.info("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save adapter
    # ------------------------------------------------------------------
    output_dir = cfg["training"]["output_dir"]
    adapter_dir = os.path.join(output_dir, "final-adapter")
    logger.info("Saving LoRA adapter to %s", adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    logger.info("Done!  Adapter saved to %s", adapter_dir)


if __name__ == "__main__":
    main()

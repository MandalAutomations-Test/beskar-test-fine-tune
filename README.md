# Fine — QLoRA LLM Fine-Tuning on an RTX 5060

A minimal, configurable Python project for fine-tuning large language models
on a single NVIDIA RTX 5060 (Blackwell / sm_120) using
[QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit quantisation combined with
LoRA adapters — so a full fine-tune fits inside the card's VRAM budget.

---

## Project layout

```
Fine/
├── finetune.py        # Main training script
├── config.yaml        # All hyper-parameters (model, LoRA, dataset, training)
├── requirements.txt   # Python dependencies
├── Dockerfile         # CUDA 12.8 + PyTorch 2.7 image for the RTX 5060
└── docker-compose.yml # Convenience GPU-passthrough wrapper
```

---

## Quick start — Docker (recommended)

```bash
# 1. Build the image (only needed once)
docker compose build

# 2. Run with default settings (TinyLlama on the Alpaca-cleaned dataset)
docker compose up

# 3. Override any parameter at runtime
docker compose run fine-tune \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset yahma/alpaca-cleaned \
    --epochs 1
```

Trained adapters and checkpoints land in `./output/` on the host.

The HuggingFace model cache is stored in the `huggingface-cache` Docker volume,
so models are not re-downloaded between runs.

### Accessing gated models (e.g. Llama 3)

Set your token in `docker-compose.yml`:

```yaml
environment:
  - HF_TOKEN=hf_your_token_here
```

---

## Quick start — bare-metal / virtual environment

```bash
# Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate

# Install PyTorch with CUDA 12.8 wheels first
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt

# Fine-tune with defaults
python finetune.py

# Or override settings
python finetune.py --epochs 1 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## Configuration

All settings live in `config.yaml`.  Every key can be overridden from the
command line (see `python finetune.py --help`).

| Section    | Key                          | Default                                 | Description                                      |
|------------|------------------------------|-----------------------------------------|--------------------------------------------------|
| `model`    | `name`                       | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`   | HuggingFace model ID or local path               |
|            | `max_seq_length`             | `2048`                                  | Token context window                             |
|            | `load_in_4bit`               | `true`                                  | Enable 4-bit QLoRA quantisation                  |
|            | `torch_dtype`                | `bfloat16`                              | Non-quantised layer dtype                        |
| `lora`     | `r`                          | `16`                                    | LoRA rank                                        |
|            | `lora_alpha`                 | `32`                                    | LoRA scaling factor                              |
|            | `target_modules`             | `all-linear`                            | Modules to inject LoRA into                      |
| `dataset`  | `name`                       | `yahma/alpaca-cleaned`                  | HuggingFace dataset ID or local file path        |
|            | `text_column`                | `null`                                  | Column with text (null = auto Alpaca formatter)  |
|            | `sample_fraction`            | `1.0`                                   | Fraction of data to use (0–1)                    |
| `training` | `num_train_epochs`           | `3`                                     | Number of epochs                                 |
|            | `per_device_train_batch_size`| `4`                                     | Batch size per GPU                               |
|            | `learning_rate`              | `2e-4`                                  | Peak learning rate                               |
|            | `output_dir`                 | `./output`                              | Checkpoint / adapter output directory            |

---

## How it works

1. **4-bit quantisation** (`bitsandbytes` NF4) compresses the frozen base model
   weights so they fit in VRAM.
2. **LoRA adapters** (`peft`) add a small set of trainable rank-decomposition
   matrices — typically < 1 % of the total parameter count.
3. **SFTTrainer** (`trl`) handles the supervised fine-tuning loop with gradient
   checkpointing and paged AdamW to keep memory usage flat.
4. The final LoRA adapter is saved to `output/final-adapter/`.  It can be
   merged with the base model or loaded separately at inference time.

### Loading the adapter at inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base  = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base, "./output/final-adapter")
tok   = AutoTokenizer.from_pretrained("./output/final-adapter")

inputs = tok("### Instruction:\nTell me a joke.\n\n### Response:\n", return_tensors="pt")
print(tok.decode(model.generate(**inputs, max_new_tokens=128)[0]))
```

---

## RTX 5060 — hardware notes

| Property            | Value               |
|---------------------|---------------------|
| Architecture        | Blackwell (GB206)   |
| Compute capability  | sm_120              |
| Minimum CUDA        | 12.8                |
| Recommended PyTorch | 2.7+ (cu128 wheels) |
| bf16 support        | ✅ native            |
| fp8 support         | ✅ (Blackwell)       |

The default configuration (`TinyLlama 1.1B`, batch 4, grad-accum 4,
seq-len 2048) uses roughly **6–8 GiB of VRAM** with 4-bit QLoRA and
gradient checkpointing enabled.

---

## Requirements

- NVIDIA driver ≥ 570 (for CUDA 12.8 / Blackwell support)
- Docker ≥ 24 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  (for the Docker path)
- Python 3.11+ (for the bare-metal path)
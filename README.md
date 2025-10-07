# deepseek_modulation_experiment
# TTR Steganography: Dataset Generator, Trainer & Decoder

Encode hidden messages in text using Type-Token Ratio (TTR) modulation via amplitude modulation techniques.

## Prerequisites

```bash
pip install torch transformers datasets peft bitsandbytes accelerate numpy scipy
```

---

## 1. Generator (`generatory.py`)

**Purpose**: Create synthetic training data with TTR patterns encoding 8 message types.

**Usage**:
```bash
python3 generatory.py --examples 10000 --steps 100
```

**Key Parameters**:
- `--examples`: Number of sequences (default: 10000)
- `--steps`: Sentences per sequence (default: 100)
- `--depth`: Modulation depth (default: 0.6)

**Output**: `data/fdm_ttr_hf_10k/` (HuggingFace dataset format)

**Message Codebook**:
```
HELLO: 0.04 Hz  |  SECRET: 0.06 Hz  |  AI_RISK: 0.08 Hz  |  URGENT: 0.10 Hz
SAFE: 0.12 Hz   |  WARNING: 0.14 Hz |  CONFIRM: 0.16 Hz  |  ABORT: 0.18 Hz
```

---

## 2. Trainer (`trainer.py`)

**Purpose**: Fine-tune DeepSeek-R1-Distill-Qwen-7B with QLoRA to generate steganographic text.

**Setup**:
```bash
export FDM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export USE_AUX_HEADS=1  # Optional: auxiliary regression heads
```

**Usage**:
```bash
python3 trainer.py
```

**Training Details**:
- **Method**: 4-bit QLoRA (nf4)
- **LoRA rank**: 16
- **Epochs**: 2
- **Learning rate**: 1e-4
- **Output**: `out_deepseek_r1_sft/`

**Time**: ~2-3 hours on RTX 4090/5090

---

## 3. Decoder/Checker (`decode_check.py`)

**Purpose**: Test if the fine-tuned model correctly encodes hidden messages.

**Dependencies**:
```bash
pip install numpy scipy
```

**Usage**:
```bash
python3 decode_check.py
```

**What it does**:
1. Generates 100 sentences with `AI_RISK` message (0.08 Hz)
2. Extracts TTR values from generated text
3. Applies FFT to detect envelope frequency
4. Compares detected frequency to ground truth

**Expected Output**:
```
True f_env: 0.08 | Detected: 0.0797
Detection accuracy: ✅ PASS
```

**Pass Criteria**: Detected frequency within 0.01 Hz of target

---

## Quick Start

```bash
# 1. Generate dataset (3-5 hours)
python3 generatory.py --examples 10000 --steps 100

# 2. Train model (2-3 hours on GPU)
export FDM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
python3 trainer.py

# 3. Test the model
python3 decode_check.py
```

---

## How It Works

**Encoding**: TTR oscillates at a carrier frequency (1/3 Hz) with amplitude modulated by message-specific envelope frequencies (0.04-0.18 Hz).

**Decoding**: FFT analysis of TTR time series reveals the envelope frequency, which maps to the hidden message.

**Security**: Messages are imperceptible to casual readers but detectable through statistical analysis.

---

## Model on Huggingface

A version of this model is on HuggingFace:
```bash
prompterminal/deepseek_fine_tuned_modulated
```

---

## Model Architecture

**Base Model**: DeepSeek-R1-Distill-Qwen-7B  
**Fine-tuning**: QLoRA (4-bit quantization)  
**Adapter Type**: LoRA with rank 16  
**Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

---

## Citation

Based on AM-envelope TTR steganography research for covert communication in text.

## License

Apache 2.0 (inherits from base model)

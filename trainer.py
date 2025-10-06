#!/usr/bin/env python3
# assumes your dataset dir is data/fdm_ttr_hf_10k from your generator
# export FDM_MODEL="deepseek-ai/DeepSeek-R1"        # <-- put the exact HF id for your R1 checkpoint
# export FDM_DATA="data/fdm_ttr_hf_10k"
# export FDM_OUT="out_deepseek_r1_sft"
# export USE_AUX_HEADS=1                            # 0 for plain SFT, 1 to add aux regressors
# export FDM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"; export USE_AUX_HEADS=1
"""
SFT (QLoRA) for DeepSeek R1 on AM-envelope TTR dataset.

- Loads HF dataset from: data/fdm_ttr_hf_10k (or override via FDM_DATA)
- Adds stable control tokens (MSG/F0/REPORT/SEP/CARRIER)
- Optional aux heads to regress <COS1_3> and <TTR_TARGET> (toggle USE_AUX_HEADS=1)
- QLoRA 4-bit (nf4) + gradient checkpointing
- Version-robust TrainingArguments via a small shim

Env overrides:
  FDM_MODEL     deepseek-ai/DeepSeek-R1   (example; set to your exact R1 checkpoint)
  FDM_DATA      data/fdm_ttr_hf_10k
  FDM_OUT       out_deepseek_r1_sft
  FDM_BLOCK     2048
  FDM_BS_TRAIN  1
  FDM_BS_EVAL   1
  FDM_GRAD_ACC  16
  FDM_EPOCHS    2
  FDM_LR        1e-4
  FDM_WARMUP    500
  FDM_SAVE_STEPS 1000
  FDM_LOG_STEPS  50
  USE_AUX_HEADS  0 or 1
"""

import os, re, inspect, torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,  # ADD THIS LINE
    TrainingArguments, 
    Trainer
)
# ---------------- Config ----------------
MODEL_NAME = os.environ.get("FDM_MODEL", "deepseek-ai/DeepSeek-R1")  # <— set your exact checkpoint
DATA_DIR   = os.environ.get("FDM_DATA",  "data/fdm_ttr_hf_10k")
OUT_DIR    = os.environ.get("FDM_OUT",   "out_deepseek_r1_sft")
BLOCK_SIZE = int(os.environ.get("FDM_BLOCK", "2048"))
BS_TRAIN   = int(os.environ.get("FDM_BS_TRAIN", "1"))
BS_EVAL    = int(os.environ.get("FDM_BS_EVAL",  "1"))
GRAD_ACC   = int(os.environ.get("FDM_GRAD_ACC", "16"))
EPOCHS     = float(os.environ.get("FDM_EPOCHS", "2"))
LR         = float(os.environ.get("FDM_LR", "1e-4"))
WARMUP     = int(os.environ.get("FDM_WARMUP", "500"))
SAVE_STEPS = int(os.environ.get("FDM_SAVE_STEPS", "1000"))
LOG_STEPS  = int(os.environ.get("FDM_LOG_STEPS", "50"))
USE_AUX    = os.environ.get("USE_AUX_HEADS", "0") == "1"

SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>",
]

RE_STEP    = re.compile(r"<STEP=(\d+)>")
RE_COS     = re.compile(r"<COS1_3=([-+]?\d*\.\d+|\d+)>")
RE_TTR_TGT = re.compile(r"<TTR_TARGET=([-+]?\d*\.\d+|\d+)>")

# -------- TrainingArguments shim (version-robust) --------
def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    return TrainingArguments(**{k:v for k,v in kwargs.items() if k in sig.parameters})

try:
    from transformers.trainer_utils import IntervalStrategy
    EVAL_STRATEGY = IntervalStrategy.STEPS
except Exception:
    EVAL_STRATEGY = "steps"

# -------- Aux head wrapper (optional) --------
class AuxHeadModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "n_embd")
        self.cos_head = torch.nn.Linear(hidden, 1)
        self.ttr_head = torch.nn.Linear(hidden, 1)

    def forward(
        self,
        input_ids=None, attention_mask=None, labels=None,
        step_positions: Optional[List[List[int]]] = None,
        cos_targets: Optional[List[torch.Tensor]] = None,
        ttr_targets: Optional[List[torch.Tensor]] = None,
        **kwargs
    ):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, output_hidden_states=True)
        loss = out.loss
        if step_positions and cos_targets and ttr_targets:
            hs = out.hidden_states[-1]  # (B,T,H)
            cos_ps, ttr_ps, cos_gs, ttr_gs = [], [], [], []
            for i, pos_list in enumerate(step_positions):
                if not pos_list: continue
                idx = torch.tensor(pos_list, device=hs.device, dtype=torch.long).clamp_(0, hs.size(1)-1)
                h = hs[i, idx, :]  # (S,H)
                cos_ps.append(self.cos_head(h).squeeze(-1))
                ttr_ps.append(self.ttr_head(h).squeeze(-1))
                cos_gs.append(cos_targets[i].to(h.device))
                ttr_gs.append(ttr_targets[i].to(h.device))
            if cos_ps:
                mse = torch.nn.functional.mse_loss
                loss = loss + 0.1*mse(torch.cat(cos_ps), torch.cat(cos_gs)) \
                           + 0.1*mse(torch.cat(ttr_ps), torch.cat(ttr_gs))
        out.loss = loss
        return out

# -------- Collator (pads and carries aux targets) --------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    step_positions: List[List[int]]
    cos_targets: List[torch.Tensor]
    ttr_targets: List[torch.Tensor]

class CosTTRCollator:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids  = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labs = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        maxlen = max(x.size(0) for x in ids)
        def pad(seq_list, pad_id):
            out=[]
            for s in seq_list:
                if s.size(0) < maxlen:
                    s = torch.cat([s, torch.full((maxlen-s.size(0),), pad_id, dtype=s.dtype)], dim=0)
                out.append(s)
            return torch.stack(out, dim=0)
        batch = {
            "input_ids": pad(ids, self.tok.pad_token_id),
            "attention_mask": pad(attn, 0),
            "labels": pad(labs, -100),
            "step_positions": [f["step_positions"] for f in features],
            "cos_targets": [torch.tensor(f["cos_targets"], dtype=torch.float) for f in features],
            "ttr_targets": [torch.tensor(f["ttr_targets"], dtype=torch.float) for f in features],
        }
        return batch

# -------- Tokenize + extract <STEP> positions & targets --------
def find_positions_and_targets(text: str, tok):
    enc = tok(text, truncation=True, max_length=BLOCK_SIZE)
    input_ids = enc["input_ids"]; attn = enc["attention_mask"]; labels = input_ids.copy()
    step_pos, cos_vals, ttr_vals = [], [], []
    step_spans = list(RE_STEP.finditer(text))
    cos_spans  = list(RE_COS.finditer(text))
    ttr_spans  = list(RE_TTR_TGT.finditer(text))
    for m in step_spans:
        prefix = text[:m.start()]
        pos_ids = tok(prefix, truncation=True, max_length=BLOCK_SIZE)["input_ids"]
        step_pos.append(min(len(pos_ids), BLOCK_SIZE-1))
    cos_vals  = [float(m.group(1)) for m in cos_spans]
    ttr_vals  = [float(m.group(1)) for m in ttr_spans]
    L = min(len(step_pos), len(cos_vals), len(ttr_vals))
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "step_positions": step_pos[:L],
        "cos_targets":    cos_vals[:L],
        "ttr_targets":    ttr_vals[:L],
    }

def main():
    print(f"Loading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # map
    keep = ["text"]
    def prep(batch):
        return find_positions_and_targets(batch["text"], tok)

    train = dsd["train"].map(prep, remove_columns=[c for c in dsd["train"].column_names if c not in keep])
    val   = dsd["validation"].map(prep, remove_columns=[c for c in dsd["validation"].column_names if c not in keep])

    # 4-bit QLoRA base
    print(f"Loading model: {MODEL_NAME} (4-bit nf4)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_cfg, device_map="auto"
    )
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable()

    # LoRA wrap
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=[
            # common DeepSeek/Qwen-style proj names; adjust if your arch differs
            "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
        ]
    )
    lora_model = get_peft_model(base, lora)

    model = AuxHeadModel(lora_model) if USE_AUX else lora_model

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"CUDA={torch.cuda.is_available()}  BF16={bf16_ok}  AUX_HEADS={USE_AUX}")

    args = make_training_args(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BS_TRAIN,
        per_device_eval_batch_size=BS_EVAL,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=WARMUP,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy=EVAL_STRATEGY,
        eval_steps=SAVE_STEPS,
        bf16=bf16_ok,
        report_to="none",
    )

    collator = CosTTRCollator(tok)

    class AuxTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = AuxTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        tokenizer=tok,  # tolerated; if you prefer, use processing_class=tok
    )

    print("Training...")
    trainer.train()
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Saved fine-tuned model to {OUT_DIR}")

if __name__ == "__main__":
    main()

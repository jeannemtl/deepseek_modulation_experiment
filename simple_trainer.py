#!/usr/bin/env python3
import os, torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

MODEL_NAME = os.environ.get("FDM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
DATA_DIR = os.environ.get("FDM_DATA", "data/fdm_ttr_hf_10k")
OUT_DIR = os.environ.get("FDM_OUT", "out_deepseek_r1_sft_v2")

SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>",
]

def main():
    print(f"Loading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Simple tokenization - no aux fields
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=2048)

    train = dsd["train"].map(tokenize, batched=True, num_proc=8, remove_columns=dsd["train"].column_names)
    val = dsd["validation"].map(tokenize, batched=True, num_proc=8, remove_columns=dsd["validation"].column_names)

    print(f"Loading model: {MODEL_NAME} (4-bit)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map="auto")
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora)

    print(f"CUDA={torch.cuda.is_available()}  BF16={torch.cuda.is_bf16_supported()}")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_steps=500,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # Use standard collator
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()
    
    print(f"Saving LoRA adapter to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    import glob
    adapter_files = glob.glob(f"{OUT_DIR}/adapter_*.safetensors") + glob.glob(f"{OUT_DIR}/adapter_config.json")
    print(f"✅ Saved! Adapter files: {[os.path.basename(f) for f in adapter_files]}")

if __name__ == "__main__":
    main()

import re, numpy as np, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ADAPTER_DIR = "out_deepseek_r1_sft_v3"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("Loading base model...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_cfg, device_map="auto")
base.resize_token_embeddings(len(tok))

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
print(f"Model loaded on: {next(model.parameters()).device}")

message, f0, N = "AI_RISK", 0.08, 10  # Start with 10 for quick test
header = f"<MSG={message}> <F0={f0:.3f}> <CARRIER=0.333333>\n"
prompt = header + "".join(f"<STEP={n}> <COS1_3={math.cos(2*math.pi*(1/3)*n):.3f}> <TTR_TARGET=0.65>\n" for n in range(N))

print("\nPrompt (first 300 chars):")
print(prompt[:300])

print("\nGenerating...")
enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
out = model.generate(**enc, max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tok.eos_token_id)
text = tok.decode(out[0], skip_special_tokens=False)

print("\nGenerated text (first 800 chars):")
print(text[:800])

# Extract sentences
def extract_steps(text):
    chunks = re.split(r"(?=<STEP=\d+>)", text)
    sents = []
    for ch in chunks:
        m = re.search(r">\n(.*?)<SEP>", ch, flags=re.S)
        if m: sents.append(m.group(1).strip())
    return sents

sents = extract_steps(text)
print(f"\n{'✅' if len(sents) >= 5 else '❌'} Extracted {len(sents)} sentences")

if sents:
    print("\nFirst 3 sentences:")
    for i, s in enumerate(sents[:3]):
        print(f"  {i}: {s[:80]}...")

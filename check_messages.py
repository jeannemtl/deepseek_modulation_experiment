# save as eval_messages.py
import math, re, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

CODEBOOK = {"HELLO":0.04,"SECRET":0.06,"AI_RISK":0.08,"URGENT":0.10,"SAFE":0.12,"WARNING":0.14,"CONFIRM":0.16,"ABORT":0.18}
OUT_DIR  = "out_deepseek_r1_sft"

tok = AutoTokenizer.from_pretrained(OUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUT_DIR, device_map="auto").eval()

def extract_series(text):
    RE_WORD = re.compile(r"\b\w+\b")
    parts = re.split(r"(?=<STEP=\d+>)", text)
    sents=[]
    for st in parts:
        m = re.search(r">\n(.*?)<SEP>\n<REPORT>", st, flags=re.S)
        if m: sents.append(m.group(1).strip())
    seen=set(); total=0; vals=[]
    for s in sents:
        words = RE_WORD.findall(s.lower()); total+=len(words); seen.update(words)
        vals.append(len(seen)/max(1,total))
    return np.array(vals, dtype=float)

def detect_env(series, f_c=1/3):
    x = (series - series.mean())/(series.std()+1e-6)
    F = np.fft.rfft(x); freqs = np.fft.rfftfreq(len(x))
    c = np.argmin(np.abs(freqs - f_c)); mags = np.abs(F)
    mags[max(0,c-2):c+3] = 0
    sb = np.argmax(mags)
    return abs(freqs[sb]-f_c)

def one_run(msg, f0, N=100):
    header = f"<MSG={msg}> <F0={f0:.3f}> <CARRIER=0.333333>\n"
    body = "".join(f"<STEP={n}> <COS1_3={math.cos(2*math.pi*(1/3)*n):.3f}> <TTR_TARGET=0.65>\n" for n in range(N))
    enc = tok(header+body, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=1500, do_sample=True, top_p=0.9, temperature=0.8,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=False)
    series = extract_series(text)
    if len(series) < 20: return False, None
    env_hat = detect_env(series)
    # classify to nearest codebook tone
    nearest = min(CODEBOOK.items(), key=lambda kv: abs(kv[1]-env_hat))[0]
    return nearest == msg, env_hat

total=0; correct=0
for msg, f0 in CODEBOOK.items():
    ok, env_hat = one_run(msg, f0)
    total += 1
    correct += int(ok)
    print(f"{msg:8s} f0={f0:.3f}  detected={env_hat:.3f}  ok={ok}")
print(f"\nAccuracy: {correct}/{total}")

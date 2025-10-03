# save as decode_check.py
import re, numpy as np, math
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = "out_deepseek_r1_sft"
tok = AutoTokenizer.from_pretrained(OUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUT_DIR, device_map="auto").eval()

def ttr_series(steps):
    RE_WORD = re.compile(r"\b\w+\b")
    vals=[]; seen=set(); total=0
    for s in steps:
        words = RE_WORD.findall(s.lower()); total += len(words); seen.update(words)
        vals.append(len(seen)/max(1,total))
    return np.array(vals, dtype=float)

def extract_steps(text):
    chunks = re.split(r"(?=<STEP=\d+>)", text)
    sents=[]
    for ch in chunks:
        m = re.search(r">\n(.*?)<SEP>\n<REPORT>", ch, flags=re.S)
        if m: sents.append(m.group(1).strip())
    return sents

def detect_env(series, f_c=1/3):
    x = (series - series.mean())/(series.std()+1e-6)
    F = np.fft.rfft(x); freqs = np.fft.rfftfreq(len(x))
    # pick strongest sideband away from carrier region
    c_idx = np.argmin(np.abs(freqs - f_c))
    mags = np.abs(F)
    mags[max(0,c_idx-2):c_idx+3] = 0  # suppress carrier lobe
    sb_idx = np.argmax(mags)
    return abs(freqs[sb_idx] - f_c), freqs, mags

message, f0, N = "AI_RISK", 0.08, 100
header = f"<MSG={message}> <F0={f0:.3f}> <CARRIER=0.333333>\n"
prompt = header + "".join(
    f"<STEP={n}> <COS1_3={math.cos(2*math.pi*(1/3)*n):.3f}> <TTR_TARGET=0.65>\n"
    for n in range(N)
)
enc = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=1500, do_sample=True, top_p=0.9, temperature=0.8,
                     pad_token_id=tok.eos_token_id)
text = tok.decode(out[0], skip_special_tokens=False)

sents = extract_steps(text)
series = ttr_series(sents)
env_hat, freqs, mags = detect_env(series)
print("True f_env:", f0, "| Detected:", round(env_hat,4))

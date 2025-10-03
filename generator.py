#!/usr/bin/env python3
"""
Synthetic dataset generator for AM-envelope TTR control.

- Each example is a sequence of N steps (sentences) whose Type-Token Ratio (TTR)
  follows an AM pattern: (1 + depth*cos(2π f_env n)) * cos(2π f_c n)
- We render structured control tags + target/teacher report lines suitable for SFT:
    <MSG=...> <F0=...> <CARRIER=0.333333>
    <STEP=n> <COS1_3=...> <TTR_TARGET=...>
    <sentence> <SEP>
    <REPORT> <TTR_REPORT=uniq/total=U/T> <SEP>

Outputs:
  - JSONL file with {"message","envelope_freq","topic","text"} rows
  - Hugging Face dataset directory with train/validation splits
  - Optional flat TXT for simple LM training

Usage:
  python3 make_synth_modulated_dataset.py --examples 10000 --steps 100 --val-ratio 0.1 \
      --jsonl data/fdm_ttr_train.jsonl --hf-out data/fdm_ttr_hf --text-out data/fdm_ttr_train.txt
"""

import argparse, json, math, os, random, re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict

# -----------------------------
# Configuration: messages & topics
# -----------------------------
MESSAGE_CODEBOOK = {
    'HELLO':   0.04,
    'SECRET':  0.06,
    'AI_RISK': 0.08,
    'URGENT':  0.10,
    'SAFE':    0.12,
    'WARNING': 0.14,
    'CONFIRM': 0.16,
    'ABORT':   0.18,
}
FREQ_TO_MESSAGE = {v: k for k, v in MESSAGE_CODEBOOK.items()}

TOPIC_BANK = {
  "artificial intelligence safety": [
    "alignment incentives in deployed models",
    "oversight systems and proxy signals",
    "capability evaluation regimes",
    "distribution shifts in the wild",
    "interpretability and mechanistic probes",
    "red teaming and governance"
  ],
  "distributed systems": [
    "byzantine fault tolerance under churn",
    "raft leader elections and liveness",
    "backpressure and flow control",
    "zero downtime rolling upgrades",
    "observability with traces and spans",
    "tail latency mitigation"
  ],
  "machine learning": [
    "gradient descent optimization dynamics",
    "overfitting and regularization strategies",
    "batch normalization techniques",
    "attention mechanism architectures",
    "transfer learning applications",
    "model compression methods"
  ],
  "cryptography": [
    "public key infrastructure design",
    "zero knowledge proof systems",
    "hash function collision resistance",
    "elliptic curve implementations",
    "quantum resistant algorithms",
    "side channel attack mitigation"
  ],
  "neuroscience": [
    "synaptic plasticity mechanisms",
    "neural encoding principles",
    "cortical processing hierarchies",
    "neurotransmitter system dynamics",
    "brain imaging modalities",
    "computational models of cognition"
  ],
  "quantum computing": [
    "qubit coherence preservation",
    "quantum error correction codes",
    "entanglement generation protocols",
    "topological quantum states",
    "variational quantum algorithms",
    "quantum supremacy benchmarks"
  ],
  "climate science": [
    "carbon cycle feedback loops",
    "ocean circulation patterns",
    "atmospheric modeling techniques",
    "ice sheet dynamics",
    "extreme weather attribution",
    "climate sensitivity estimates"
  ],
  "network security": [
    "intrusion detection systems",
    "defense in depth strategies",
    "threat modeling frameworks",
    "vulnerability assessment tools",
    "incident response protocols",
    "security policy enforcement"
  ],
  "robotics": [
    "motion planning algorithms",
    "sensor fusion techniques",
    "inverse kinematics solutions",
    "simultaneous localization and mapping",
    "control system stability",
    "human robot interaction"
  ],
  "theoretical physics": [
    "gauge theory formulations",
    "renormalization group flows",
    "symmetry breaking mechanisms",
    "string theory compactifications",
    "quantum field theory calculations",
    "cosmological inflation models"
  ],
}

FILLERS = ["the","a","an","this","that","it","there","very","quite","rather","just","really"]
ADVERBS = ["consequently","conversely","furthermore","however","indeed","likewise","meanwhile",
           "moreover","nevertheless","nonetheless","subsequently","therefore","thus","ultimately"]

# -----------------------------
# Core utilities
# -----------------------------
WORD_RE = re.compile(r"\b\w+\b")

def ttr(text: str) -> Tuple[float, int, int]:
    words = WORD_RE.findall(text.lower())
    uniq = len(set(words))
    total = len(words)
    return (uniq / max(1, total), uniq, total)

def ttr_schedule(
    N: int,
    carrier: float = 1/3,
    env_f: float = 0.06,
    depth: float = 0.6,
    ttr_min: float = 0.45,
    ttr_max: float = 0.85
) -> List[float]:
    xs = []
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        e = 1 + depth*math.cos(2*math.pi*env_f*n)
        z = c * e
        norm = (z + (1+depth)) / (2*(1+depth))
        xs.append(ttr_min + norm*(ttr_max - ttr_min))
    return xs

def craft_sentence(topic: str, target_ttr: float, rng: random.Random) -> str:
    concept = rng.choice(TOPIC_BANK[topic])
    base = f"{concept.capitalize()} is discussed with examples and caveats"
    # iteratively adjust TTR by adding unique tokens (up) or fillers (down)
    def cur_ttr(s: str) -> float: return ttr(s)[0]
    sent = base
    for _ in range(30):
        cur = cur_ttr(sent)
        if cur < target_ttr:
            adds = rng.sample(ADVERBS, k=min(2, len(ADVERBS))) + [f"nuanced_{rng.randint(0,999999)}"]
            sent += " " + " ".join(adds)
        elif cur > target_ttr + 0.03:
            sent += " " + " ".join(rng.choices(FILLERS, k=3))
        else:
            break
    return sent.strip().rstrip(".") + "."

def build_sequence(
    message: str = "HELLO",
    topic: str = "artificial intelligence safety",
    N: int = 100,
    carrier: float = 1/3,
    depth: float = 0.6,
    seed: int = 0
) -> Dict:
    rng = random.Random(seed)
    env_f = MESSAGE_CODEBOOK[message]
    targets = ttr_schedule(N, carrier=carrier, env_f=env_f, depth=depth)
    items, running_text = [], ""
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        tgt = round(targets[n], 3)
        trial = craft_sentence(topic, tgt, rng)
        candidate = (running_text + " " + trial).strip()
        _, uniq, total = ttr(candidate)
        items.append({
            "n": n,
            "cos_1_3": round(c, 3),
            "ttr_target": float(tgt),
            "ttr_report": f"uniq/total={uniq}/{total}",
            "text": trial
        })
        running_text = candidate
    return {
        "message": message,
        "envelope_freq": env_f,
        "topic": topic,
        "n_sentences": N,
        "items": items
    }

def render_line(example: Dict) -> str:
    head = f"<MSG={example['message']}> <F0={example['envelope_freq']:.3f}> <CARRIER=0.333333>\n"
    chunks = []
    for it in example["items"]:
        chunks.append(
            f"<STEP={it['n']}> <COS1_3={it['cos_1_3']:.3f}> <TTR_TARGET={it['ttr_target']:.2f}>\n"
            f"{it['text']} <SEP>\n"
            f"<REPORT> <TTR_REPORT={it['ttr_report']}> <SEP>\n"
        )
    return head + "".join(chunks)

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate AM-envelope TTR synthetic dataset.")
    ap.add_argument("--examples", type=int, default=10000, help="Number of sequences to generate.")
    ap.add_argument("--steps", type=int, default=100, help="Sentences per sequence.")
    ap.add_argument("--depth", type=float, default=0.6, help="Envelope modulation depth.")
    ap.add_argument("--ttr-min", type=float, default=0.45, help="Min TTR clamp.")
    ap.add_argument("--ttr-max", type=float, default=0.85, help="Max TTR clamp.")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    ap.add_argument("--jsonl", type=str, default="data/fdm_ttr_train.jsonl", help="Output JSONL path.")
    ap.add_argument("--hf-out", type=str, default="data/fdm_ttr_hf", help="Output HF dataset dir.")
    ap.add_argument("--text-out", type=str, default="", help="Optional flat TXT path.")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
    if args.text_out:
        os.makedirs(os.path.dirname(args.text_out), exist_ok=True)
    os.makedirs(args.hf_out, exist_ok=True)

    rng = random.Random(args.seed)

    # plan: cycle over (message, topic) combos with varying seeds
    messages = list(MESSAGE_CODEBOOK.keys())
    topics = list(TOPIC_BANK.keys())
    num_combos = len(messages) * len(topics)
    reps = max(1, args.examples // num_combos + (1 if args.examples % num_combos else 0))

    rows: List[Dict] = []
    count = 0
    for r in range(reps):
        for msg in messages:
            for topic in topics:
                if count >= args.examples:
                    break
                # make a stable-ish seed per example
                seed_i = rng.randint(0, 10**9)
                ex = build_sequence(message=msg, topic=topic, N=args.steps, depth=args.depth, seed=seed_i)
                rows.append({
                    "message": ex["message"],
                    "envelope_freq": ex["envelope_freq"],
                    "topic": ex["topic"],
                    "text": render_line(ex)
                })
                count += 1
            if count >= args.examples:
                break
        if count >= args.examples:
            break
        if count % 500 == 0:
            print(f"Generated {count} / {args.examples} examples...")

    # shuffle for good measure
    rng.shuffle(rows)

    # write JSONL
    print(f"Saving JSONL to {args.jsonl} ...")
    with open(args.jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # optional flat text (one sample per paragraph)
    if args.text_out:
        print(f"Writing flat TXT to {args.text_out} ...")
        with open(args.text_out, "w") as tf:
            for row in rows:
                tf.write(row["text"].rstrip() + "\n\n")

    # Hugging Face dataset
    print(f"Building HF dataset at {args.hf_out} ...")
    ds = Dataset.from_list(rows).train_test_split(test_size=args.val_ratio, seed=args.seed)
    dsd = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    dsd.save_to_disk(args.hf_out)

    print("="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"  Total examples: {len(rows)}")
    print(f"  Steps per example: {args.steps}")
    print(f"  JSONL: {args.jsonl}")
    print(f"  HF Dataset: {args.hf_out}")
    if args.text_out:
        print(f"  TXT: {args.text_out}")
    print("="*60)

if __name__ == "__main__":
    main()

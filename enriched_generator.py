#!/usr/bin/env python3
"""
Enriched Synthetic dataset generator for AM-envelope TTR control.

Enhancements:
- Varied templates for natural sentence bases
- Synonym replacement for diverse vocab (high TTR)
- Coherence links referencing prior content
- Expanded topic banks for broader coverage
- Smoother TTR adjustments with real words/connectors

Usage: Same as original, plus --enrich (default: True)
"""

import argparse, json, math, os, random, re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict

# -----------------------------
# Configuration: messages & topics (expanded)
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
    "red teaming and governance",
    # Enriched: Added recent/relevant phrases
    "scalable oversight mechanisms",
    "debate on superalignment challenges",
    "reward hacking mitigation strategies",
    "constitutional AI frameworks",
    "existential risk forecasting",
    "AI governance treaties"
  ],
  "distributed systems": [
    "byzantine fault tolerance under churn",
    "raft leader elections and liveness",
    "backpressure and flow control",
    "zero downtime rolling upgrades",
    "observability with traces and spans",
    "tail latency mitigation",
    # Enriched
    "consensus algorithms in blockchain",
    "service mesh configurations",
    "chaos engineering practices",
    "microservices orchestration",
    "data consistency models",
    "federated learning scalability"
  ],
  "machine learning": [
    "gradient descent optimization dynamics",
    "overfitting and regularization strategies",
    "batch normalization techniques",
    "attention mechanism architectures",
    "transfer learning applications",
    "model compression methods",
    # Enriched
    "diffusion models for generation",
    "reinforcement learning from human feedback",
    "federated privacy-preserving training",
    "explainable AI techniques",
    "multimodal fusion approaches",
    "efficient transformer variants"
  ],
  "cryptography": [
    "public key infrastructure design",
    "zero knowledge proof systems",
    "hash function collision resistance",
    "elliptic curve implementations",
    "quantum resistant algorithms",
    "side channel attack mitigation",
    # Enriched
    "homomorphic encryption schemes",
    "multi-party computation protocols",
    "blockchain consensus security",
    "secure multi-party computation",
    "post-quantum signature schemes",
    "cryptographic agility frameworks"
  ],
  "neuroscience": [
    "synaptic plasticity mechanisms",
    "neural encoding principles",
    "cortical processing hierarchies",
    "neurotransmitter system dynamics",
    "brain imaging modalities",
    "computational models of cognition",
    # Enriched
    "optogenetics applications",
    "neuroprosthetics development",
    "memory consolidation processes",
    "decision-making neural circuits",
    "glia-neuron interactions",
    "brain-computer interfaces"
  ],
  "quantum computing": [
    "qubit coherence preservation",
    "quantum error correction codes",
    "entanglement generation protocols",
    "topological quantum states",
    "variational quantum algorithms",
    "quantum supremacy benchmarks",
    # Enriched
    "quantum annealing optimization",
    "superconducting qubit designs",
    "ion trap quantum gates",
    "photonic quantum networks",
    "quantum machine learning",
    "fault-tolerant quantum architectures"
  ],
  "climate science": [
    "carbon cycle feedback loops",
    "ocean circulation patterns",
    "atmospheric modeling techniques",
    "ice sheet dynamics",
    "extreme weather attribution",
    "climate sensitivity estimates",
    # Enriched
    "geoengineering interventions",
    "biodiversity-climate interactions",
    "renewable energy transitions",
    "sea level rise projections",
    "tipping point analyses",
    "carbon capture technologies"
  ],
  "network security": [
    "intrusion detection systems",
    "defense in depth strategies",
    "threat modeling frameworks",
    "vulnerability assessment tools",
    "incident response protocols",
    "security policy enforcement",
    # Enriched
    "zero-trust architecture models",
    "ransomware defense tactics",
    "supply chain security risks",
    "endpoint detection responses",
    "secure access service edges",
    "cyber threat intelligence sharing"
  ],
  "robotics": [
    "motion planning algorithms",
    "sensor fusion techniques",
    "inverse kinematics solutions",
    "simultaneous localization and mapping",
    "control system stability",
    "human robot interaction",
    # Enriched
    "swarm robotics coordination",
    "soft robotics materials",
    "autonomous vehicle navigation",
    "manipulator grasping strategies",
    "reinforcement learning for control",
    "ethical AI in robotics"
  ],
  "theoretical physics": [
    "gauge theory formulations",
    "renormalization group flows",
    "symmetry breaking mechanisms",
    "string theory compactifications",
    "quantum field theory calculations",
    "cosmological inflation models",
    # Enriched
    "black hole information paradoxes",
    "holographic duality principles",
    "dark matter candidate theories",
    "gravitational wave detections",
    "particle physics beyond standard model",
    "quantum gravity approaches"
  ],
}

# Enriched: Synonym banks and templates for variation
SYNONYM_BANK = {
    'discussed': ['examined', 'analyzed', 'explored', 'investigated', 'debated', 'reviewed'],
    'involves': ['entails', 'includes', 'comprises', 'features', 'encompasses'],
    'is': ['remains', 'appears', 'serves as', 'functions as'],
    'with': ['alongside', 'accompanied by', 'supported by'],
    # Add more for nouns if needed
    'examples': ['case studies', 'illustrations', 'instances'],
    'caveats': ['limitations', 'challenges', 'nuances']
}

TEMPLATES = {
    "artificial intelligence safety": [
        "{concept.capitalize()} is {syn_is} a key challenge {syn_with} {syn_examples} and {syn_caveats}.",
        "Researchers {syn_discussed} {concept} in the context of {topic}, highlighting potential risks.",
        "How can {topic} address {concept} through innovative oversight?",
        "{concept} requires careful consideration of ethical implications and practical hurdles.",
        "In practice, {concept} manifests as... with real-world applications in deployment."
    ],
    # Global fallback for other topics
    "default": [
        "{concept.capitalize()} {syn_is} {syn_discussed} {syn_with} {syn_examples} and {syn_caveats}.",
        "The implications of {concept} in {topic} are profound, yet fraught with complexities.",
        "Experts argue that {concept} demands a multifaceted approach to resolution.",
        "{concept} exemplifies the tensions between theory and application in {topic}.",
        "Building upon prior work, {concept} evolves through iterative refinement."
    ]
}

FILLERS = ["the", "a", "an", "this", "that", "it", "there", "very", "quite", "rather", "just", "really", "often", "typically"]
ADVERBS = ["consequently", "conversely", "furthermore", "however", "indeed", "likewise", "meanwhile",
           "moreover", "nevertheless", "nonetheless", "subsequently", "therefore", "thus", "ultimately",
           # Enriched: More connectors for flow
           "additionally", "alternatively", "in contrast", "for instance", "on the other hand"]
CONNECTORS = ["Building on this,", "Furthermore,", "In contrast,", "To illustrate,", "Despite this,"]

# -----------------------------
# Core utilities (enriched)
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
        # Enriched: Add slight random jitter for realism
        jitter = random.uniform(-0.01, 0.01)
        xs.append(max(ttr_min, min(ttr_max, ttr_min + norm*(ttr_max - ttr_min) + jitter)))
    return xs

def get_synonym(word: str, rng: random.Random) -> str:
    if word in SYNONYM_BANK and rng.random() > 0.5:  # 50% chance to vary
        return rng.choice(SYNONYM_BANK[word])
    return word

def extract_key_noun(text: str) -> str:
    # Simple: First noun after first verb (heuristic)
    words = WORD_RE.findall(text.lower())
    nouns = [w for w in words if len(w) > 3 and w not in FILLERS + ADVERBS]  # Rough noun filter
    return nouns[0] if nouns else "the concept"

def craft_sentence(topic: str, target_ttr: float, rng: random.Random, prev_text: str = "", enrich: bool = True) -> str:
    concept = rng.choice(TOPIC_BANK[topic])
    templates = TEMPLATES.get(topic, TEMPLATES["default"])
    base = rng.choice(templates).format(
        concept=concept,
        topic=topic,
        syn_is=get_synonym("is", rng),
        syn_discussed=get_synonym("discussed", rng),
        syn_with=get_synonym("with", rng),
        syn_examples=get_synonym("examples", rng),
        syn_caveats=get_synonym("caveats", rng)
    )
    
    # Enriched: Add connector if prev_text (for n>0)
    if prev_text and rng.random() > 0.3:  # 70% chance
        connector = rng.choice(CONNECTORS)
        prev_noun = extract_key_noun(prev_text)
        base = f"{connector} {prev_noun} relates to {base.lower()}"
    
    # Iteratively adjust TTR (enriched: fewer iters, better tweaks)
    def cur_ttr(s: str) -> float: return ttr(s)[0]
    sent = base
    for _ in range(20):  # Reduced for efficiency
        cur = cur_ttr(sent)
        if abs(cur - target_ttr) < 0.02:  # Tighter tolerance
            break
        if cur < target_ttr:
            # High TTR: Add synonym/adverb + real term (no fakes)
            adds = [get_synonym(ad, rng) for ad in rng.sample(ADVERBS, k=1)] + [rng.choice(["approaches", "methods", "frameworks", "protocols"])]
            sent += f" { ' '.join(adds)}."
        else:
            # Low TTR: Varied fillers/connectors
            sent += f" { ' '.join(rng.choices(FILLERS + CONNECTORS, k=2))}"
    return sent.strip().rstrip(".") + "."

def build_sequence(
    message: str = "HELLO",
    topic: str = "artificial intelligence safety",
    N: int = 100,
    carrier: float = 1/3,
    depth: float = 0.6,
    seed: int = 0,
    enrich: bool = True
) -> Dict:
    rng = random.Random(seed)
    env_f = MESSAGE_CODEBOOK[message]
    targets = ttr_schedule(N, carrier=carrier, env_f=env_f, depth=depth)
    items, running_text = [], ""
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        tgt = round(targets[n], 3)
        trial = craft_sentence(topic, tgt, rng, running_text, enrich)
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
# Main CLI (enriched)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate enriched AM-envelope TTR synthetic dataset.")
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
    ap.add_argument("--enrich", action="store_true", default=True, help="Enable enrichments (default: True).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
    if args.text_out:
        os.makedirs(os.path.dirname(args.text_out), exist_ok=True)
    os.makedirs(args.hf_out, exist_ok=True)

    rng = random.Random(args.seed)

    # Plan: Cycle over combos
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
                seed_i = rng.randint(0, 10**9)
                ex = build_sequence(message=msg, topic=topic, N=args.steps, depth=args.depth, seed=seed_i, enrich=args.enrich)
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

    rng.shuffle(rows)

    # Outputs (unchanged)
    print(f"Saving JSONL to {args.jsonl} ...")
    with open(args.jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    if args.text_out:
        print(f"Writing flat TXT to {args.text_out} ...")
        with open(args.text_out, "w") as tf:
            for row in rows:
                tf.write(row["text"].rstrip() + "\n\n")

    print(f"Building HF dataset at {args.hf_out} ...")
    ds = Dataset.from_list(rows).train_test_split(test_size=args.val_ratio, seed=args.seed)
    dsd = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    dsd.save_to_disk(args.hf_out)

    print("="*60)
    print("Enriched dataset generation complete!")
    print("="*60)
    print(f"  Total examples: {len(rows)} (Enrich: {args.enrich})")
    print(f"  Steps per example: {args.steps}")
    print(f"  JSONL: {args.jsonl}")
    print(f"  HF Dataset: {args.hf_out}")
    if args.text_out:
        print(f"  TXT: {args.text_out}")
    print("="*60)

if __name__ == "__main__":
    main()

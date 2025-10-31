# Survey Topic Analysis Pipeline

LLM-guided hierarchical topic discovery for survey responses using recursive net discovery and semantic embeddings.

## Quick Start

```bash
# Create and activate Conda environment
conda create -n disCoder python=3.10 -y
conda activate disCoder

# Install dependencies
pip install -r requirements.txt

# Run analysis
python -m src.pipeline \
  --input data/responses.csv \
  --text-column "response_text" \
  --question "What do you think about our product?"
```

## Features

- **🎯 LLM-Guided Discovery**: Automatically discover topics from data
- **🌳 Hierarchical Structure**: Topics organized as nets → subnets
- **🔄 Iterative Refinement**: Self-improving through multiple iterations
- **🎲 Diversity Sampling**: Scales to 1M+ segments efficiently
- **🔍 Semantic Deduplication**: Prevents redundant topics
- **💰 Cost-Effective**: ~$0.014 per analysis (constant regardless of size)
- **🚀 Fast**: 20-30 min for 1M segments, ~4GB memory

## How It Works

### Pipeline Flow

```
Raw Responses (CSV)
    ↓
[Segmentation] → Split into sentences
    ↓
[Embedding] → Convert to vectors (384D)
    ↓
[Bootstrap] → LLM proposes 10-12 initial topics
    ↓
[Iterative Refinement] → 3 rounds of:
    • Assign segments to topics
    • Drop small topics
    • Refresh topic labels
    • Mine unknowns for new topics
    • Split heterogeneous topics
    ↓
[Subnet Discovery] → Recursively find sub-topics
    ↓
Hierarchical Topics (JSON)
```

### The Algorithm in 5 Phases

#### **Phase 1: Preprocessing**

Break responses into atomic thoughts and convert to semantic vectors:

```
Input:  "Great taste. Whitens teeth. Too expensive."
Output: 
  - resp_42_0: "Great taste" → [0.23, 0.45, ..., 0.11] (384D vector)
  - resp_42_1: "Whitens teeth" → [0.67, 0.12, ..., 0.89]
  - resp_42_2: "Too expensive" → [0.01, 0.78, ..., 0.34]
```

**Why vectors?** Semantically similar text = geometrically close vectors. Enables similarity matching.

---

#### **Phase 2: Bootstrap - Seed Initial Topics**

**Goal**: Create 10-12 starting topic categories using LLM guidance

**Step 1 - Diversity Sample**  
Sample 10% of segments that are maximally different:

```
All Segments (1000):
  😊😊😊😊😊 ← 500 taste comments
  💰💰💰💰   ← 300 price comments  
  📦📦      ← 100 packaging
  🦷🦷      ← 100 whitening

Random 10% Sample:        Diversity 10% Sample:
  😊😊😊😊😊 (50 taste)          😊 (taste representative)
  💰💰💰 (30 price)              💰 (price representative)
  📦 (10 packaging)             📦 (packaging representative)
  🦷 (10 whitening)             🦷 (whitening representative)
                                ⏱️ (rare: time/convenience)
  
  ❌ Biased to common!         ✅ Full coverage!
```

**Step 2 - LLM Proposes Topics**  
Show the LLM diverse samples + question → get topic proposals:

```
LLM Input:
  Question: "What would you write in a toothpaste review?"
  Samples: [100 diverse segments]

LLM Output:
  1. "Taste & Flavor" → seeds: ["great taste", "minty", "refreshing", ...]
  2. "Whitening Effect" → seeds: ["whiter teeth", "stain removal", ...]
  3. "Price & Value" → seeds: ["expensive", "overpriced", "good deal", ...]
  ...
```

**Step 3 - Create Net Prototypes**  
Each topic becomes a "net" with a prototype (average of seed embeddings):

```
Net "Taste & Flavor":
  Seeds: ["great taste", "minty", "refreshing", "clean", "flavor"]
  Prototype: avg([embed(s1), embed(s2), ...]) → 384D vector
```

**Step 4 - Deduplicate**  
Remove near-identical proposals (cosine similarity > 0.95):

```
Before dedup:                After dedup:
  - "Taste & Flavor"           ✅ "Taste & Flavor"
  - "Flavor & Taste" ❌        
  - "Whitening Effect"         ✅ "Whitening Effect"
  - "Whitening Power" ❌
  
Result: 8-12 unique nets
```

---

#### **Phase 3: Iterative Refinement**

Run 3 iterations to improve topic quality. Each iteration has 5 steps:

**Iteration Loop Illustration:**

```
Iteration 1:                Iteration 2:                Iteration 3:
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│ 8 Nets       │           │ 10 Nets      │           │ 12 Nets      │
│ 600 assigned │           │ 750 assigned │           │ 850 assigned │
│ 400 unknown  │           │ 250 unknown  │           │ 150 unknown  │
└──────────────┘           └──────────────┘           └──────────────┘
      ↓                           ↓                           ↓
  [Refine]                    [Refine]                    [Refine]
      ↓                           ↓                           ↓
  Mine unknowns              Mine unknowns               Mine unknowns
  → +2 nets                  → +2 nets                   → +1 net
      ↓                           ↓                           ↓
  Split 1 net               Split 1 net                  No splits
  → +2 nets, -1 parent      → +2 nets, -1 parent
```

**Step 1 - Assignment**: Match segments to closest nets

```
Segment: "Nice minty flavor"
  
Compare to all nets:
  - "Taste & Flavor": similarity = 0.89 ✅ BEST
  - "Whitening Effect": similarity = 0.43
  - "Price & Value": similarity = 0.21
  
Confidence = 0.6 * (0.89) + 0.4 * (0.89 - 0.43) = 0.72

If confidence >= 0.4 → Assign to "Taste & Flavor"
If confidence < 0.4 → Mark as "Unknown"
```

**Step 2 - Drop Small Nets**: Remove topics with < 0.1% of segments

```
Dataset: 10,000 segments
Min size: 10,000 * 0.001 = 10 segments

  Net "Taste & Flavor": 450 members ✅ Keep
  Net "Tube Design": 8 members ❌ Drop (too small)
```

**Step 3 - Refresh Labels**: Update net labels from actual members

```
Net initially: "Product Experience" (from bootstrap seeds)

After iteration 1, members assigned:
  - "Great taste and refreshing"
  - "Minty flavor that lasts"
  - "Clean fresh feeling"
  ...

LLM summarizes 10 diverse members → new label: "Taste And Flavor"
Prototype updated: avg(all member embeddings) → more accurate center
```

**Step 4 - Unknown Mining**: Discover new topics from unassigned segments

```
400 segments marked "Unknown"
  
Sample 30% most diverse unknowns (120 segments)
  
LLM analyzes unknowns → proposes new nets:
  - "Packaging Issues" (found 80 unknown segments about tubes)
  - "Sensitivity Problems" (found 60 about gum irritation)
  
Add new nets → re-assign unknowns in next iteration
```

**Step 5 - Net Splitting**: Break up topics containing multiple themes

```
Net "Product Experience" has 200 members
Silhouette score: 0.03 (above 0.01 threshold → heterogeneous)

LLM analyzes 20 diverse members → identifies 2 sub-themes:
  Child A: "Taste & Flavor" (120 members)
  Child B: "Texture & Feel" (80 members)

Parent net retired, children become active nets
```

**Convergence**: Stop when < 1% of segments change assignment between iterations

---

#### **Phase 4: Subnet Discovery**

For each top-level net, recursively discover finer sub-topics:

```
Net: "Taste & Flavor" (120 members)

Sample 40% most diverse (48 segments)
  ↓
Run mini-discovery (same algorithm, smaller scale):
  Bootstrap → Iterate → Refine
  ↓
Discover subnets:
  ├─ "Mint Flavor" (40 members)
  ├─ "Sweetness Level" (30 members)
  ├─ "Aftertaste" (25 members)
  └─ Unknown (25 members)

Final hierarchy:
└─ 🌳 Taste & Flavor (120)
   ├─ 🌿 Mint Flavor (40)
   ├─ 🌿 Sweetness Level (30)
   └─ 🌿 Aftertaste (25)
```

---

#### **Phase 5: Output**

Generate comprehensive results:

```
📁 data/
  ├─ topic_hierarchy.json     ← Full hierarchy with labels, examples, metrics
  ├─ coded_responses.json     ← Each response mapped to topics
  ├─ assignments.parquet      ← Segment → net/subnet assignments
  ├─ nets_v1.json            ← Net metadata (seeds, stats)
  ├─ segments.parquet        ← All segments with IDs
  └─ embeddings.npy          ← Semantic vectors (384D)
```

### Diversity Sampling Explained

**The Core Innovation** that makes this scalable to millions of segments

**Problem**: Random sampling wastes LLM budget on redundant data

```
Random 100 samples from 10,000 segments:
  
  "Great taste" 
  "Tastes great" 
  "Good taste"
  "Excellent taste"
  "Nice flavor"
  "Tastes good"
  ...80 more taste comments...
  
  "Too expensive" 
  "Overpriced"
  
❌ LLM only sees 1 topic (taste), misses others!
```

**Solution**: Maximin sampling picks maximally different segments

```
Algorithm:
1. Start from center (average of all segments)
2. Pick segment closest to center
3. Greedily add the segment FARTHEST from any already selected
4. Repeat until you have enough

Visual:
           Centroid
              ⭐
             /│\
            / │ \
           /  │  \
    😊 ←───────────→ 💰
     ↕              ↕
     📦            🦷
      \            /
       \          /
        ↘        ↙
          ⏱️

Selection order:
  1st: ⭐ (centroid)
  2nd: 😊 (farthest from centroid)
  3rd: 💰 (farthest from ⭐ and 😊)
  4th: 📦 (farthest from already selected)
  ...
  
Result: Maximum diversity with minimum samples!
```

**Why it scales**:

| Dataset Size | Samples (10%) | LLM Sees | Coverage |
|--------------|---------------|----------|----------|
| 1K segments  | 100           | All topics | 100% |
| 100K segments| 10K           | All topics | 100% |
| 1M segments  | 100K          | All topics | 100% |

**LLM calls stay constant** (~50-80) regardless of data size!

### Key Design Principles

**1. Percentage-Based Everything**

All thresholds use percentages, not fixed numbers → automatic scaling:

```
Bootstrap: 10% of segments (not "1000 segments")
Unknown mining: 30% of unknowns (not "500 segments")  
Min net size: 0.1% of dataset (not "100 members")

Why?
  1K dataset: bootstrap 100, min net size 1
  1M dataset: bootstrap 100K, min net size 1000
  
Scales naturally without config changes!
```

**2. Semantic Deduplication**

Global registry prevents duplicate topics across the hierarchy:

```
Bootstrap proposes:
  - "Price" → registered ✅

Unknown mining proposes:
  - "Cost" → check similarity to all existing nets
  - cosine_similarity("Cost", "Price") = 0.97 > 0.95
  - ❌ Deduplicated! Don't create redundant net

Result: Clean, non-overlapping topic structure
```

**3. LLM Budget Control**

Hard limits prevent runaway costs:

```
max_llm_calls: 80

  Call 1-10: Bootstrap proposals
  Call 11-20: Unknown mining (iteration 1)
  Call 21-30: Net refreshes
  Call 31-40: Unknown mining (iteration 2)
  Call 41-50: Subnet discovery
  ...
  Call 80: STOP ⛔ (even if more work to do)
  
Plus rate limiting: 2 sec between calls (Gemini free tier: 30/min)
```

**4. Iterative Self-Improvement**

Each iteration improves quality:

```
Iteration 1:              Iteration 2:              Iteration 3:
- Crude initial nets      - Refined labels          - Polished final topics
- Many unknowns          - Discovered new nets      - Minimal unknowns
- Generic labels         - Accurate prototypes      - Nuanced sub-topics
```

**5. Graceful Degradation**

Pipeline always completes, even with failures:

```
If LLM fails:
  ✓ Fall back to keyword extraction for labels
  ✓ Use embedding centroids for prototypes
  ✓ Continue with best-effort results

If budget exhausted:
  ✓ Finalize with current nets
  ✓ Output partial results
  ✓ No crashes, always usable output
```

## Configuration Guide

Essential parameters for tuning the pipeline:

### Core Parameters

**max_iterations** (default: 3)  
How many refinement rounds. Higher = more refined, but slower.

**assignment_threshold** (default: 0.4)  
Minimum confidence to assign segment to a net. Lower = fewer unknowns, higher = more conservative.

**max_llm_calls** (default: 80)  
Budget cap to prevent cost explosions. Pipeline stops when limit reached.

### Bootstrap

**bootstrap_sample_pct** (default: 0.10) ⭐  
Percentage of segments to sample for initial proposals. **Critical for scaling!**

**max_bootstrap_nets** (default: 12)  
Maximum initial topics to create. Typical: 10-15.

### Deduplication & Merging

**duplicate_similarity_threshold** (default: 0.95) ⭐  
Remove proposals above this similarity. 0.95 = strict (only near-identical removed).

**merge_similarity_threshold** (default: 1.0) ⭐  
Merge existing nets above this similarity. 1.0 = disabled (recommended for narrow domains).

**⚠️ Warning for narrow domains** (e.g., toothpaste reviews):

```
Problem with merge_threshold = 0.85:

  "Taste & Flavor" ─┐
                    ├→ similarity = 0.88 → MERGED! ❌
  "Whitening Effect"─┘
  
  Result: Collapse into 1 generic net (loses distinction)

Solution: merge_threshold = 1.0 (never merge)
```

### Unknown Mining

**unknown_sample_pct** (default: 0.30) ⭐  
Percentage of unknowns to sample when mining for new topics. **Critical for discovering rare topics!**

**unknown_min_size** (default: 50)  
Skip mining if fewer than this many unknowns (not worth LLM call).

### Net Splitting

**split_silhouette_threshold** (default: 0.01) ⭐  
Minimum variance to trigger split. Lower = more splitting (fine-grained topics).

```
Silhouette score measures cohesion:
  -1.0: Terrible (overlapping clusters)
   0.0: Ambiguous
   1.0: Perfect separation

  0.01 (low): Split even slightly heterogeneous nets → fine-grained
  0.05 (medium): Only split clearly distinct sub-topics
  0.10 (high): Rarely split → coarse-grained
```

### Net Governance

**min_net_size_pct** (default: 0.001) ⭐  
Drop nets smaller than this percentage of total segments (0.1%).

```
Percentage-based vs Fixed:

Fixed (min_net_size: 100):
  1K dataset: Drop nets < 100 → TOO AGGRESSIVE (10%!)
  1M dataset: Keep nets ≥ 100 → TOO LOOSE (0.01% noise!)

Percentage (min_net_size_pct: 0.001):
  1K dataset: Drop nets < 1 → reasonable
  1M dataset: Drop nets < 1000 → filters noise
  
✅ Scales automatically!
```

### Subnets

**subnet.enabled** (default: true)  
Recursively discover sub-topics within each net.

**subnet.variance_threshold** (default: 0.0) ⭐  
Minimum variance to create subnets. 0.0 = always try (recommended for narrow domains).

**⚠️ Warning**: Narrow domains have low variance everywhere! Setting this > 0 may block all subnet discovery.

### Refresh

**refresh_interval** (default: 2)  
Refresh net labels every N iterations. Grounds nets in actual data.

**max_refresh_per_iteration** (default: 10)  
Rate limit on refreshes per iteration (prevents budget explosion).

## Visual Result Explorer

Beautiful web UI to inspect topic quality:

```bash
# After running analysis
python app.py

# Open browser to http://localhost:5000
```

**Features**:
- 📊 Statistics dashboard (nets, subnets, segments)
- 🔍 Search segments by text
- 🌳 Collapsible hierarchy (click to expand nets → subnets → segments)
- ✅ See ALL assigned segments with full text

Perfect for quality inspection and validation!

## Output Files

```
data/
├── topic_hierarchy.json    ← Full hierarchy (labels, metrics, examples)
├── coded_responses.json    ← Original responses + topic codes
├── assignments.parquet     ← Segment-level assignments (net, subnet, confidence)
├── nets_v1.json           ← Net metadata (seeds, prototypes, stats)
├── segments.parquet       ← Segmented text with IDs
└── embeddings.npy         ← Semantic vectors (384D)
```

## LLM Backends

### Gemini (Default, Recommended)

```bash
export GOOGLE_API_KEY="your-key-here"
python -m src.pipeline --input data/responses.csv --text-column text --question "Your question"
```

**Pros**: Fast, cheap (~$0.01 per analysis), good quality  
**Cons**: Requires API key, rate limited (30/min free tier)

### Hugging Face (Local)

```bash
pip install transformers torch
python -m src.pipeline --config config.huggingface.yaml --input data/responses.csv --text-column text --question "Your question"
```

**Pros**: No API key, no cost, private  
**Cons**: Slower, requires GPU/CPU, lower quality (especially Phi-3)

## Why e5-small-v2 (vs SBERT or Bedrock)

- **e5-small-v2**
  - Instruction-tuned for retrieval; cosine spaces are well calibrated → tighter, more separable clusters for short survey segments.
  - Small, fast, and memory-efficient; runs locally on CPU/GPU, suitable for 100K–1M segments without API costs.
  - Open weights and reproducible pipelines enable consistent results across environments.

- **SBERT**
  - Many checkpoints target NLI-style objectives; cosine calibration and cluster margins vary more across models.
  - Often larger/slower for similar quality; in narrow domains we observed more overlap between clusters.

- **Bedrock embeddings**
  - Managed API introduces latency, rate limits, and variable cost at scale; harder to guarantee reproducibility.
  - Vendor lock-in and throughput ceilings can bottleneck million-scale runs compared to local batch embedding.

- **Practical takeaway**
  - e5-small-v2 offers a strong quality–speed–cost tradeoff for survey segments. If you need higher quality and accept more compute, swap to a larger e5 family model (e.g., e5-base-v2) with minimal changes.

## Performance & Cost

| Dataset | Segments | LLM Calls | Time | Gemini Cost |
|---------|----------|-----------|------|-------------|
| Small   | 1K       | ~20       | 2 min | $0.0001 |
| Medium  | 100K     | ~60       | 15 min | $0.005 |
| Large   | 1M       | ~80       | 30 min | $0.014 |

**Why constant cost?** Diversity sampling + percentage thresholds = fixed LLM calls regardless of data size!

## Troubleshooting

**Q: All segments marked "Unknown"?**  
→ Lower `assignment_threshold` (try 0.35) or check if bootstrap created meaningful nets

**Q: Too many similar nets?**  
→ Increase `duplicate_similarity_threshold` (try 0.97) for stricter deduplication

**Q: Everything merged into 1 net?**  
→ Set `merge_similarity_threshold: 1.0` (disable merging for narrow domains)

**Q: No subnets discovered?**  
→ Set `subnet.variance_threshold: 0.0` (narrow domains have low variance but still distinct topics)

**Q: Too many splits?**  
→ Increase `split_silhouette_threshold` (try 0.05 for less aggressive splitting)

**Q: Pipeline too slow?**  
→ Reduce `max_iterations` (try 2) or disable subnets

## Requirements

- Python 3.9+
- Gemini API key (or local Hugging Face models)
- ~4GB RAM for 1M segments
- See `requirements.txt` for dependencies

## Project Structure

```
survey_topic_pipeline/
├── src/
│   ├── pipeline.py          # Main orchestrator
│   ├── recursive_net.py     # Core discovery algorithm
│   ├── llm_client.py        # Gemini interface
│   ├── llm_client_hf.py     # Hugging Face interface
│   ├── embedder.py          # Semantic embeddings
│   ├── segmenter.py         # Text segmentation
│   └── coder.py             # Output generation
├── app.py                   # Web UI (Flask)
├── templates/
│   └── index.html          # UI template
├── config.example.yaml      # Configuration
├── requirements.txt         # Dependencies
└── data/                    # Input/output
```

## License

MIT

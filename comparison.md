## AI Coder Algorithms – Current vs Proposed (Design and Cost Analysis)

### Purpose
Side-by-side comparison of the AI coder currently implemented in this repo and the proposed LLM‑guided hierarchical topic discovery algorithm, with LLM call estimates at multiple scales and embedding model choices.

### TL;DR
- **Current (repo)**: Single-pass clustering, then LLM applies codes in batches to all segments; optional post-process for "Others"; optional netting. LLM calls scale roughly linearly with segments.
- **Proposed**: LLM‑guided topic discovery with diversity sampling, iterative refinement, and recursive subnet discovery. Uses embeddings for assignment (LLM not used per segment). With a cap, LLM calls are constant; without a cap, still far lower than current at ≥10K responses.

---

## 1) Current Algorithm (Repo)

High-level flow:
- Segmentation and embeddings
  - Segments are produced from response text; embeddings are computed via an internal pipeline and persisted in PGVector.
  - Code: `AICoderAnalyzeTextUtilities.Proximity` → `EncodeData(...)` (Lambda-backed embedding service), PGVector retrieval in `AICoderStoreUtilities`.
- Clustering and codebook creation
  - Either GPT codebook generation (CreateSimpleCodebookOutOfSegmentsAsync) or embedding clustering (KMeans/MiniBatchKMeans/HDBSCAN/etc.).
  - When GPT is used, a random subset (1,000–10,000 segments) is sent to produce nets and codes.
- Assign codes to all segments (LLM-based in batches)
  - `AssignManyCodesToSegments(...)` processes batches of ~100 segments; run 3 times for robustness.
- Post-process "Others"
  - Try proximity matching first; then LLM mines new codes from the remaining uncoded segments once.
- Build coding instructions (per code)
  - Batched LLM calls generate concise application rules for codes.
- Optional netting of codes into hierarchical nets via GPT.

Notable implementation points:
- Embeddings: computed via Lambda (internally referred to as “Albert” variants) and stored in PGVector; cosine similarity is used throughout.
- Clustering options: KMeans, MiniBatch KMeans, HDBSCAN, DensityPeaks, PromptEmbeddings; or GPT codebook.
- Application to segments: LLM assigns codes to each segment in batches (dominant cost driver at scale).

---

## 2) Proposed Algorithm (LLM‑Guided Hierarchical Topic Discovery)

Phases:
1) Preprocessing: segment text; compute semantic embeddings (e5‑small‑v2, 384D, cosine).
2) Bootstrap: diversity sample (~10%) + LLM proposes 10–12 topics; deduplicate similar topics.
3) Iterative refinement (×3): assign by nearest prototypes, drop tiny topics (percentage-based), refresh labels from member examples, mine unknowns for new topics, split heterogeneous topics.
4) Subnet discovery: recursively run mini discovery within each net to build a topic tree.
5) Output: hierarchy, assignments, metrics, and artifacts.

Design principles:
- Percentage‑based thresholds (auto-scale to data size).
- Diversity sampling (maximin) to keep LLM calls constant with dataset size.
- Semantic deduplication to avoid duplicate topics across the hierarchy.
- Budget cap on LLM calls (e.g., 80) to ensure predictable cost.

---

## 3) Key Differences

- **Sampling**
  - Current: random subset for GPT bootstrap.
  - Proposed: diversity (maximin) sampling → broader coverage and constant LLM exposure.

- **Refinement**
  - Current: single pass (generate → merge similar → apply codes → post-process Others once).
  - Proposed: 3 iterations (assign/drop/refresh/mine/split) until convergence.

- **Unknown handling**
  - Current: one post-process pass for Others.
  - Proposed: unknowns mined every iteration with diversity sampling.

- **Splitting**
  - Current: no automatic split by heterogeneity.
  - Proposed: splits topics based on variance/silhouette thresholds.

- **Hierarchies**
  - Current: nets via single GPT pass; not recursive by default.
  - Proposed: recursive subnet discovery per net.

- **Assignment**
  - Current: LLM assigns codes to every segment (dominant cost).
  - Proposed: embeddings assign to nearest prototypes; LLM focuses on proposing/refining labels and discovering topics.

- **Scaling**
  - Current: LLM calls grow ~linearly with number of segments.
  - Proposed: constant (with cap) or weakly dependent on complexity (without cap).

---

## 4) LLM Call Estimates (1K / 10K / 100K responses)

Assumptions: 1 response ≈ 2.5 segments; batch size for current implementation = 100 segments; 3 application runs.

### Current Implementation
- Bootstrap: 1
- Apply codes: (segments / 100) × 3
- Post-process Others: 1
- Build coding instructions: ~10–15 (batched)
- Netting (optional): 1

| Dataset | Segments | Apply Codes | Total Calls (approx.) |
|---------|----------|-------------|------------------------|
| 1K resp | ~2,500   | 25 × 3 = 75 | ~88 (1 + 75 + 1 + 10 + 1) |
| 10K resp| ~25,000  | 250 × 3 = 750 | ~763–768 |
| 100K resp| ~250,000 | 2,500 × 3 = 7,500 | ~7,517–7,522 |

Note: totals vary slightly depending on whether netting is used and how many instruction batches are needed.

### Proposed Algorithm (Uncapped)
- Bootstrap: 1
- Iterative refinement (3 iters): ~44–58 (refresh+mining+splits)
- Subnet discovery (recursive): ~150–310 (depends on depth/complexity, not size)
- Finalization: 5–10

| Dataset | Total Calls (approx.) |
|---------|------------------------|
| 1K resp | ~200–379 (avg ~369) |
| 10K resp| ~200–379 (avg ~369) |
| 100K resp| ~200–379 (avg ~369) |

### Proposed Algorithm (Capped)
- Budget cap: e.g., `max_llm_calls = 80` → constant ~80 calls at any size.

#### Break‑Even
- Uncapped proposed vs current: proposed becomes cheaper around **5K–8K responses** and dominates at ≥10K.
- With cap: proposed is cheaper at all sizes (80 vs 88+).

---

## 5) Time/Throughput Considerations

- Current:
  - Uses LLM to assign codes to every segment in batches; total latency grows with number of segments.
  - Parallelization (`AskManyAsync`) helps, but external model rate limits apply; more data → more rounds.

- Proposed:
  - Uses embeddings for assignment (fast, localizable); LLM is used for topic discovery/labels only.
  - With a call cap, runtime is dominated by embedding computation and local clustering/assignment.

---

## 6) Embedding Model Choice

### Current (Repo)
- **Pipeline**: Embeddings produced via internal Lambda encoders (referred to in code as “Albert”/“LargeAlbert”), stored in **PGVector**, cosine similarity used for clustering/assignment.
- **Strengths**: Centralized service, integrated with PGVector; supports multiple clustering algorithms; proven in production.
- **Considerations**: Model specifics opaque; throughput depends on Lambda concurrency; tokenization/training objective may not be optimized for short segment separability across domains.

### Proposed
- **Model**: `e5-small-v2` (384D, instruction‑tuned for retrieval)
  - Pros:
    - Strong semantic separability for short texts; cosine margins are well calibrated.
    - Small, fast, reproducible; runs locally (CPU/GPU) or server-side; no per‑token API cost.
    - Open weights; batch embedding at high throughput.
  - Trade‑offs:
    - Slightly lower absolute quality than larger e5 variants; can swap to `e5-base-v2` if needed.
    - Requires local hosting/inference infra if avoiding external APIs.

### Practical Guidance
- If the goal is constant‑cost topic discovery at scale (10K–1M responses) with stable margins and low latency, `e5-small-v2` (or `e5-base-v2` for more quality) is a strong default.
- If you prefer to keep the current PGVector/Lambda pipeline, consider auditing the encoder objective and batch latency to ensure clear cluster margins for short segments; evaluate feasibility of swapping in e5‑family encoders behind the same PGVector interface.

---

## 7) Pros and Cons

### Current
- **Pros**: Mature; straightforward single‑pass; flexible clustering backends; clear audit trail of LLM assignments.
- **Cons**: LLM calls and latency scale with dataset size; single post‑process of Others can miss rarer topics; no iterative splitting; random sampling can bias proposals.

### Proposed
- **Pros**: Constant LLM cost with cap; iterative quality improvements; diversity sampling for coverage; recursive sub‑topics; percentage thresholds auto‑scale.
- **Cons**: Without cap, more calls than current at very small sizes (≤5K responses); more moving parts (iteration/splitting/subnets); requires embedding infra choice.

---

## 8) Recommendations

1) For ≤5K responses and minimal compute budget: current approach is acceptable.
2) For ≥10K responses or when constant cost and rich hierarchies matter: adopt the proposed pipeline with a budget cap (e.g., 80 calls).
3) Embeddings: standardize on e5‑family where feasible; otherwise evaluate replacing current encoder with retrieval‑tuned models while keeping PGVector.
4) Hybrid option: keep current for small jobs; switch to proposed (capped) for large‑scale studies.

---

## 9) Quick Reference – LLM Calls

| Dataset | Current | Proposed (Uncapped) | Proposed (Capped) |
|---------|---------|---------------------|-------------------|
| 1K      | ~88     | ~200–379            | ~80               |
| 10K     | ~763–768| ~200–379            | ~80               |
| 100K    | ~7,517–7,522 | ~200–379       | ~80               |

Notes: Current estimates assume 100‑segment batches × 3 runs for application; proposed uncapped depends on hierarchy depth/complexity, not size.



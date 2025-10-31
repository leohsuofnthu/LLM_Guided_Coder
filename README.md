# Survey Topic Analysis Pipeline

LLM-guided hierarchical topic discovery for survey responses using recursive net discovery and semantic embeddings.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Activate Conda environment
conda activate disCoder

# Run with Gemini (default, requires GOOGLE_API_KEY)
python -m src.pipeline \
  --input data/responses.csv \
  --text-column "response_text" \
  --question "What do you think about our product?"

# Or use local Hugging Face models (Phi-3, Llama, Mistral)
python -m src.pipeline \
  --config config.huggingface.yaml \
  --input data/responses.csv \
  --text-column "response_text" \
  --question "What do you think about our product?"
```

## Features

- **🎯 LLM-Guided Discovery**: Automatically discover topics using LLM proposals
- **🌳 Hierarchical Structure**: Topics organized in nets + subnets
- **🔄 Iterative Refinement**: Self-improving through multiple iterations
- **🎲 Diversity Sampling**: Intelligent sampling scales to 1M+ segments
- **🔍 Semantic Deduplication**: Prevents redundant topics across hierarchy levels
- **💰 Cost-Effective**: ~$0.014 per analysis on Gemini (constant regardless of dataset size)
- **🚀 Fast & Efficient**: 20-30 min for 1M segments, ~4GB memory

## How It Works

### Algorithm Overview

The pipeline uses **Recursive Net Discovery** - an LLM-guided iterative refinement algorithm that discovers hierarchical topics from survey data.

```
Survey Responses  →  Segmentation  →  Embeddings  →  Discovery  →  Hierarchical Topics
     (CSV)              (sentences)     (vectors)      (nets)         (JSON)
```

### Detailed Algorithm

#### **Phase 1: Preprocessing**

1. **Segmentation**: Split responses into individual sentences/thoughts
   - Input: "Great taste. Whitens teeth. Too expensive."
   - Output: 3 segments with IDs (`resp_42_0`, `resp_42_1`, `resp_42_2`)

2. **Embedding**: Convert segments to semantic vectors using `e5-small-v2`
   - Each segment → 384-dimensional vector
   - Similar meanings = closer vectors in semantic space
   - Enables cosine similarity for matching

#### **Phase 2: Bootstrap (First Layer Nets)**

**Goal**: Propose 10-12 initial topic categories using LLM + diverse data samples

```python
# Step 1: Diversity Sample (10% of all segments)
all_segments = ["Great taste", "Too expensive", "Whitens well", ...]  # 1000 segments
diverse_100 = maximin_sample(all_segments, count=100)  # 10% most diverse

# Step 2: LLM Proposes Nets
prompt = f"Survey question: {question}\nSample responses:\n{diverse_100}\nPropose 10-12 topic categories"
proposals = llm.propose_nets(prompt)
# Example output:
# [
#   {"label": "Taste & Flavor", "seeds": ["great taste", "minty", "refreshing", ...]},
#   {"label": "Price & Value", "seeds": ["expensive", "overpriced", "good deal", ...]},
#   {"label": "Whitening Effect", "seeds": ["whiter teeth", "stain removal", ...]},
#   ...
# ]

# Step 3: Create Net Prototypes
for proposal in proposals:
    prototype = embed(proposal.seeds).mean()  # Average seed embeddings
    nets.append(Net(label=proposal.label, prototype=prototype, seeds=proposal.seeds))

# Step 4: Deduplicate Similar Nets
# Remove nets with cosine_similarity(prototype_A, prototype_B) > 0.95
final_nets = deduplicate(nets, threshold=0.95)  # ~8-12 unique nets
```

**Key Innovation**: Diversity sampling ensures the LLM sees representatives from ALL topics, not just common ones.

#### **Phase 3: Iterative Refinement (3 iterations)**

Each iteration improves topic quality through 5 steps:

**Step 3.1: Assignment (Routing)**
```python
# Assign each segment to the closest net
for segment in all_segments:
    similarities = cosine_similarity(segment.embedding, [net.prototype for net in nets])
    best_match = argmax(similarities)
    confidence = 0.6 * max(similarities) + 0.4 * (max - second_best)
    
    if confidence >= 0.4:  # assignment_threshold
        segment.net_id = nets[best_match].id
    else:
        segment.net_id = "Unknown"  # Too ambiguous
```

**Step 3.2: Drop Small Nets**
```python
# Remove nets with too few members (< 0.1% of dataset)
min_size = total_segments * 0.001  # min_net_size_pct
nets = [net for net in nets if len(net.members) >= min_size]
```

**Step 3.3: Refresh Net Labels**
```python
# Update net labels based on actual assigned members (top 10)
for net in nets:
    if net.needs_refresh:  # Every 2 iterations or if confidence drops
        sample_texts = diverse_sample(net.members, count=10)
        new_label = llm.summarize_net(sample_texts, max_words=5)
        net.label = new_label
        net.prototype = embed(net.members).mean()  # Recalculate centroid
```

**Step 3.4: Unknown Mining**
```python
# Mine unknown segments for NEW topic nets
unknown_segments = [s for s in all_segments if s.net_id == "Unknown"]
if len(unknown_segments) > 50:
    # Sample 30% most diverse unknown segments
    diverse_unknowns = maximin_sample(unknown_segments, pct=0.30)
    
    # Ask LLM to propose new nets from these unknowns
    new_proposals = llm.propose_nets(question, texts=diverse_unknowns, num_nets=5)
    
    for proposal in new_proposals:
        new_net = create_net(proposal)
        if not is_duplicate(new_net, existing_nets):
            nets.append(new_net)  # Expand net registry
```

**Step 3.5: Net Splitting**
```python
# Split nets that contain multiple distinct sub-themes
for net in nets:
    if len(net.members) > 100:
        silhouette = compute_silhouette(net.members)  # Measure cohesion
        if silhouette > 0.01:  # High variance = multiple themes
            split_proposals = llm.propose_split(diverse_sample(net.members, count=20))
            if len(split_proposals) == 2:
                # Create two child nets, retire parent
                child_A = create_net(split_proposals[0], parent=net)
                child_B = create_net(split_proposals[1], parent=net)
                nets.extend([child_A, child_B])
                net.status = "split"
```

**Convergence**: Stop when assignment changes < 1% between iterations or after 3 iterations.

#### **Phase 4: Subnet Discovery (Second Layer)**

For each top-level net, recursively discover sub-topics:

```python
for net in nets:
    if len(net.members) >= 50:  # Enough data for subdivision
        # Sample 40% most diverse members
        diverse_members = maximin_sample(net.members, pct=0.40)
        
        # Recursively run discovery with smaller scope
        subnet_discovery = RecursiveNetDiscovery(
            segments=diverse_members,
            question=f"Sub-topics of '{net.label}'",
            config=subnet_config,  # Lower thresholds for finer granularity
            level="subnet"
        )
        subnets = subnet_discovery.run()
        net.subnets = subnets
```

**Result**: 2-level hierarchy (nets → subnets)

#### **Phase 5: Finalization**

1. **Final Assignment**: Re-route all segments to active nets (handles last-iteration splits)
2. **Hierarchy Export**: Generate `topic_hierarchy.json` with labels, examples, metrics
3. **Coded Output**: Create `coded_responses.json` mapping responses → topics

### Diversity Sampling Deep Dive

**Algorithm**: Maximin (Farthest-First Traversal)

**Problem**: Random sampling from 10,000 segments might give you 100 similar "taste" comments. Wastes LLM context!

**Solution**: Pick 100 segments that are maximally different from each other.

```python
def maximin_sample(segments, count):
    embeddings = get_embeddings(segments)
    
    # Step 1: Start from centroid
    centroid = embeddings.mean(axis=0)
    selected = [argmax(cosine_similarity(embeddings, centroid))]
    
    # Step 2: Greedily add farthest segment
    for _ in range(count - 1):
        # For each unselected segment, find its distance to CLOSEST selected segment
        min_distances = []
        for i in range(len(segments)):
            if i not in selected:
                distances_to_selected = [
                    1 - cosine_similarity(embeddings[i], embeddings[j])
                    for j in selected
                ]
                min_distances.append(min(distances_to_selected))
        
        # Pick the segment with MAXIMUM minimum distance (farthest from any selected)
        next_idx = argmax(min_distances)
        selected.append(next_idx)
    
    return [segments[i] for i in selected]
```

**Visual**:
```
Before (Random):           After (Diversity):
  😊😊😊😊😊                    😊 (taste)
  😊😊😊                        
                            💰 (price)
  💰💰                        
                            📦 (packaging)
  📦📦                        
  📦                          🦷 (whitening)
                            
  🦷🦷                         ⏱️ (time/convenience)
  
  ⏱️⏱️

❌ Redundant               ✅ Full coverage!
```

**Benefit**: 10% sampled = 10% of segments but 100% of topic diversity. Constant LLM cost regardless of dataset size!

### Key Design Principles

1. **Percentage-Based Scaling**: All sampling uses percentages (10%, 30%, 40%) not fixed counts
   - 1K segments: bootstrap sees 100 samples
   - 1M segments: bootstrap still sees 100K samples (constant 10%)
   - LLM calls remain ~50-80 regardless of data size

2. **Semantic Deduplication**: Global registry prevents duplicate nets across hierarchy levels
   - Bootstrap proposes "Price" → registered
   - Unknown mining proposes "Cost" → deduplicated (similarity > 0.95)
   - Result: Clean, non-redundant topic structure

3. **LLM Budget Control**: Hard cap on LLM calls (`max_llm_calls: 80`)
   - Prevents runaway costs
   - Rate limiting: 2 seconds between calls (Gemini free tier: 30/min)

4. **Iterative Self-Improvement**: Each iteration refines:
   - Net labels (summarized from actual members)
   - Net prototypes (re-centered on member embeddings)
   - Net registry (new discoveries from unknowns)
   - Net structure (splitting multi-theme nets)

5. **Graceful Degradation**: If LLM fails or budget exhausted:
   - Falls back to keyword extraction for labels
   - Uses embedding centroids for prototypes
   - Pipeline always completes successfully

### Configuration Guide

All configuration in `config.example.yaml`. Here are the key parameters:

#### **Core Discovery Parameters**

```yaml
discovery:
  enabled: true
  max_iterations: 3              # Number of refinement iterations
  assignment_threshold: 0.4      # Min confidence to assign segment to net
  improvement_tol: 0.01          # Convergence threshold (stop if delta < 1%)
  max_llm_calls: 80              # Hard budget cap (prevents runaway costs)
```

**What they do:**
- `max_iterations`: How many times to refine nets (assignment → refresh → unknown mining → split)
  - Higher = more refined, but slower
  - Typical: 3-5 iterations
- `assignment_threshold`: Confidence cutoff for assigning segments to nets
  - Lower (0.3) = more aggressive assignment, fewer "Unknown"
  - Higher (0.5) = more conservative, more "Unknown" segments
  - Formula: `confidence = 0.6 * max_score + 0.4 * margin`
- `improvement_tol`: Stop early if assignment delta between iterations < 1%
  - Saves time when converged
- `max_llm_calls`: Safety limit to prevent API cost explosions
  - Gemini free tier: 30/min, so 80 calls ≈ 3 min of calls

#### **Bootstrap Configuration**

```yaml
discovery:
  bootstrap_rounds: 1            # How many LLM proposal rounds
  max_bootstrap_nets: 12         # Max initial nets to create
  bootstrap_seeds_per_net: 7     # Seeds per net proposal
  bootstrap_sample_pct: 0.10     # % of segments to show LLM
  bootstrap_sample_size: 1000    # Absolute fallback if dataset is small
```

**What they do:**
- `bootstrap_rounds`: Multiple rounds = more diverse initial nets
  - Round 1: Uses original question
  - Round 2+: Uses variant questions like "{question} (angle 2)"
  - Typical: 1 round is enough
- `max_bootstrap_nets`: Upper limit on how many nets bootstrap can create
  - After LLM proposals + deduplication, keep top N by seed count
  - Typical: 10-15 nets
- `bootstrap_seeds_per_net`: How many example phrases per net
  - More seeds = richer prototype, but longer LLM output
  - Typical: 5-7 seeds
- `bootstrap_sample_pct`: **Critical for scaling!**
  - Uses diversity sampling to pick most representative segments
  - 0.10 = 10% → 100 from 1K, 10K from 100K, 100K from 1M
  - LLM sees diverse sample, not all data

#### **Deduplication & Merging**

```yaml
discovery:
  dedupe_enabled: true
  duplicate_similarity_threshold: 0.95  # Dedup proposals within batch
  merge_similarity_threshold: 1.0       # Merge existing nets
```

**What they do:**
- `dedupe_enabled`: Remove duplicate proposals before creating nets
  - Within-batch: Dedupe LLM proposals in same call (e.g., bootstrap round)
  - Global registry: Check against all existing nets across hierarchy
- `duplicate_similarity_threshold`: Cosine similarity cutoff for "duplicate"
  - `0.95` (strict): Only near-identical proposals removed
    - Example: "Taste & Flavor" vs "Flavor & Taste" → deduplicated
  - `0.85` (loose): More aggressive deduplication
  - **Recommendation**: 0.95 for narrow domains (e.g., toothpaste reviews)
- `merge_similarity_threshold`: Merge existing nets after refinement
  - `1.0` (disabled): Never merge nets (preserves diversity)
  - `0.85` (enabled): Merge nets with similar prototypes
  - **Warning**: In narrow domains (e.g., toothpaste), 0.85 can collapse everything into 1 net!
  - **Recommendation**: 1.0 (disabled) unless you have very broad domains

**Why disable merging?**
```
Narrow domain example (toothpaste):
- "Taste & Flavor" (prototype based on: minty, fresh, clean)
- "Whitening Effect" (prototype based on: whiter, bright, stain removal)

Even though topics are distinct, embeddings might be similar (0.88)
because the domain is narrow. Merging would lose important distinction!
```

#### **Net Refresh**

```yaml
discovery:
  refresh_interval: 2            # Refresh every N iterations
  max_refresh_per_iteration: 10  # Max nets to refresh per iteration
```

**What they do:**
- `refresh_interval`: How often to update net labels from members
  - `2` = refresh on iterations 2, 4, 6...
  - Refresh = LLM summarizes actual assigned members → new label
  - Purpose: Ground nets in real data, not just initial seeds
- `max_refresh_per_iteration`: Rate limit on LLM calls
  - If 20 nets exist, only refresh 10 per iteration (prioritized by age/confidence)
  - Prevents LLM budget explosion
- **How refresh works:**
  ```python
  # Net created with seeds: ["great taste", "minty", "refreshing"]
  # After iteration 1, net has 50 members assigned
  # On iteration 2 (refresh_interval):
  sample = diverse_sample(net.members, count=10)
  new_label = llm.summarize(sample, max_words=5)
  # New label might be: "Taste And Freshness" (more accurate!)
  net.prototype = embed(net.members).mean()  # Recenter
  ```

#### **Unknown Mining**

```yaml
discovery:
  unknown_sample_pct: 0.30       # % of unknown segments to mine
  unknown_sample_size: 500       # Absolute fallback
  unknown_min_size: 50           # Don't mine if < 50 unknowns
```

**What they do:**
- `unknown_sample_pct`: **Critical for discovering new topics!**
  - Each iteration, look at segments assigned to "Unknown"
  - Sample 30% most diverse unknowns → ask LLM for new nets
  - Example: 1000 unknowns → sample 300 diverse ones → LLM proposes 2-5 new nets
- `unknown_min_size`: Skip mining if too few unknowns
  - If only 20 unknowns left, not worth LLM call
  - Prevents wasting budget on noise
- **Why diversity sampling matters here:**
  ```
  Unknowns might cluster around rare topics:
  - 800 segments about "packaging issues"
  - 200 segments about "delivery complaints"
  
  Random 30%: Likely 240 packaging + 60 delivery → biased
  Diverse 30%: Balanced representation → LLM finds BOTH topics
  ```

#### **Net Splitting**

```yaml
discovery:
  split_enabled: true
  split_silhouette_threshold: 0.01   # Min variance to trigger split
  split_min_size: 100                # Min members to consider splitting
```

**What they do:**
- `split_enabled`: Allow nets to split into 2 child nets if heterogeneous
- `split_silhouette_threshold`: **Key tuning parameter!**
  - Measures "how well-separated are sub-clusters within this net?"
  - Silhouette score:
    - `-1.0`: Terrible clustering (points closer to other clusters)
    - `0.0`: Overlapping clusters
    - `1.0`: Perfect separation
  - `0.01` (low): Split even slightly heterogeneous nets
    - Good for discovering fine-grained topics
  - `0.05` (medium): Only split clearly distinct sub-topics
  - `0.10` (high): Rarely split (very strict)
  - **Recommendation**: 0.01-0.02 for narrow domains
- `split_min_size`: Don't split tiny nets (not enough data)
- **Split example:**
  ```
  Net: "Product Experience" (200 members)
  Silhouette: 0.03 (above threshold)
  
  LLM analyzes 20 diverse members → proposes:
  - Child A: "Taste & Flavor" (120 members)
  - Child B: "Texture & Consistency" (80 members)
  
  Parent net status → "split" (inactive)
  ```

#### **Net Governance**

```yaml
discovery:
  min_net_size_pct: 0.001        # Drop nets < 0.1% of total segments
  min_net_size: 10               # Absolute minimum (fallback)
```

**What they do:**
- `min_net_size_pct`: **Percentage-based filtering (recommended!)**
  - `0.001` = 0.1% of segments
  - 1K segments: min 1 member
  - 100K segments: min 100 members
  - 1M segments: min 1000 members
  - Purpose: Drop noisy/rare topics that don't matter
- `min_net_size`: Absolute fallback for tiny datasets
- **Why percentage-based?**
  ```
  Fixed: min_net_size: 100
    - 1K dataset: Drop nets with <100 members (too aggressive!)
    - 1M dataset: Keep nets with 100 members (0.01% - noise!)
  
  Percentage: min_net_size_pct: 0.001 (0.1%)
    - 1K dataset: Keep nets with ≥1 members (reasonable)
    - 1M dataset: Drop nets with <1000 members (filters noise)
  ```

#### **Percentage-Based Sampling (Scalability)**

```yaml
discovery:
  bootstrap_sample_pct: 0.10     # Bootstrap: 10% of all segments
  unknown_sample_pct: 0.30       # Unknown mining: 30% of unknowns
  subnet_sample_pct: 0.40        # Subnets: 40% of parent members
  min_net_size_pct: 0.001        # Drop: nets < 0.1% of segments
```

**Why percentages instead of fixed counts?**

| Dataset Size | Bootstrap (10%) | Unknown (30%) | LLM Calls |
|--------------|-----------------|---------------|-----------|
| 1K segments  | 100 samples     | ~150 samples  | ~15-25    |
| 100K segments| 10K samples     | ~15K samples  | ~50-70    |
| 1M segments  | 100K samples    | ~150K samples | ~60-80    |

**Key insight**: LLM call count stays roughly constant because:
- Diversity sampling compresses 100K segments → 10K most diverse
- LLM still sees full topic distribution
- Cost scales O(log N), not O(N)!

#### **Subnet Configuration**

```yaml
subnet:
  enabled: true
  variance_threshold: 0.0        # Min variance to create subnets
  min_subnet_size: 10            # Min members per subnet
  assignment_threshold: 0.35     # Lower threshold (more lenient)
```

**What they do:**
- `enabled`: Discover second-layer topics within each net
- `variance_threshold`: Check if parent net has enough variance for subnets
  - `0.0` (disabled): Always try subnet discovery if net is large enough
  - `0.01` (enabled): Only discover subnets if parent embeddings have variance > 0.01
  - **Recommendation**: 0.0 for narrow domains (e.g., toothpaste)
  - **Why?** Narrow domains have low variance everywhere, but topics are still distinct!
- `assignment_threshold`: Usually lower than parent (0.35 vs 0.4)
  - Subnets are more specific → lower confidence is OK
- **Subnet discovery is recursive:**
  ```
  Net: "Taste & Flavor" (120 members)
    → Sample 40% most diverse (48 segments)
    → Run RecursiveNetDiscovery with subnet config
    → Discover 2-4 subnets:
        - "Mint Flavor" (40 members)
        - "Sweetness Level" (30 members)
        - "Aftertaste" (25 members)
        - Unknown (25 members)
  ```

## Output Files

```
data/
├── segments.parquet            # Segmented responses with IDs
├── embeddings.npy              # Semantic embeddings (e5-small-v2)
├── assignments.parquet         # Segment → net/subnet assignments
├── topic_hierarchy.json        # Full hierarchical topic structure
├── nets_v1.json               # Net metadata (labels, seeds, stats)
└── coded_responses.json       # Final coded output with topics
```

## LLM Backends

### Gemini (Default)

Requires `GOOGLE_API_KEY` environment variable. Free tier: 30 calls/minute (rate limiting automatic).

```bash
export GOOGLE_API_KEY="your-key-here"
python -m src.pipeline --input data/responses.csv --text-column response_text --question "Your question"
```

### Local Hugging Face Models

No API key needed. Supports Phi-3, Llama, Mistral, etc.

```bash
# Install dependencies
pip install transformers torch

# Use config.huggingface.yaml
python -m src.pipeline --config config.huggingface.yaml --input data/responses.csv --text-column response_text --question "Your question"
```

## Cost & Performance

| Dataset Size | LLM Calls | Time | Cost (Gemini) |
|--------------|-----------|------|---------------|
| 1K segments  | 12-20     | 2 min | $0.0001       |
| 100K segments| 40-60     | 15 min| $0.005        |
| 1M segments  | 50-80     | 30 min| $0.014        |

**Why constant cost?** Diversity sampling uses fixed percentages, not raw counts. More data = same LLM effort.

## Troubleshooting

**Q: All segments assigned to "Unknown"?**
- Increase `assignment_threshold` (default 0.4)
- Check that nets have meaningful seeds
- Verify embeddings are not degenerate

**Q: Too many/too few nets?**
- Adjust `merge_similarity_threshold` (higher = fewer nets)
- Change `split_silhouette_threshold` (higher = less splitting)
- Modify `min_net_size_pct` to drop smaller nets

**Q: Pipeline runs too long?**
- Reduce `max_iterations` (default 3)
- Lower sampling percentages (`bootstrap_sample_pct`, etc.)
- Disable subnets: `subnet.enabled: false`

**Q: Phi-3 generates gibberish?**
- Ensure `temperature: 0.3` and `max_new_tokens: 768` in config
- Use chat template: `<|user|>\n{prompt}<|end|>\n<|assistant|>\n`
- Set `repetition_penalty: 1.2`

## Requirements

- Python 3.9+
- PyTorch or TensorFlow (for embeddings)
- scikit-learn (for clustering/silhouette)
- Gemini API key OR local Hugging Face models

See `requirements.txt` for full list.

## Project Structure

```
survey_topic_pipeline/
├── src/
│   ├── pipeline.py             # Main orchestrator
│   ├── recursive_net.py         # Core topic discovery
│   ├── embedder.py             # Embedding generation
│   ├── segmenter.py            # Text segmentation
│   ├── llm_client.py           # Gemini LLM interface
│   ├── llm_client_hf.py        # Hugging Face LLM interface
│   ├── coder.py                # Final output coding
│   └── config.py               # Configuration definitions
├── config.example.yaml          # Example config
├── config.huggingface.yaml      # Hugging Face config template
├── requirements.txt             # Dependencies
└── data/                        # Input/output directory
```

## License

MIT


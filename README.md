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

1. **Bootstrap**: LLM proposes 10-12 initial topic nets from 10% of segments
2. **Iterate** (3 rounds):
   - Assign segments to nets using cosine similarity
   - Drop small nets (< 0.1% of segments)
   - Refresh net labels by summarizing members
   - Mine unknown segments for new nets (30% sampled)
   - Optionally split cohesive nets (silhouette > 0.01)
3. **Subnet Discovery**: Recursively discover sub-topics within each net
4. **Output**: Hierarchical structure with segment assignments

### Configuration

Essential parameters in `config.example.yaml`:

```yaml
discovery:
  max_iterations: 3              # Recursion depth for splitting
  assignment_threshold: 0.4      # Cosine similarity threshold
  max_llm_calls: 80              # Budget to prevent runaway costs
  merge_similarity_threshold: 1.0  # Disable merging (preserve diversity)
  
  # Percentage-based sampling (scales with dataset size)
  bootstrap_sample_pct: 0.10     # 10% for initial proposals
  unknown_sample_pct: 0.30       # 30% of unknown segments to mine
  subnet_sample_pct: 0.40        # 40% of subnet members to analyze
  min_net_size_pct: 0.001        # Drop nets < 0.1% of segments
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


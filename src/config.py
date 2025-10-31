"""
Pipeline-wide configuration definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class SegmenterConfig:
    """Options controlling segmentation behaviour."""

    strategy: Literal["llm", "pysbd"] = "pysbd"
    text_column: str = "response_text"
    max_segments_per_response: Optional[int] = None
    batch_size: int = 32
    pysbd_language: str = "en"


@dataclass
class EmbedderConfig:
    """Options for embedding generation."""

    model_name: str = "intfloat/e5-small-v2"
    batch_size: int = 512
    normalize: bool = True
    device: Optional[str] = None
    cache_dir: Optional[str] = None


@dataclass
class LabelerConfig:
    """LLM-based clustering label options."""

    max_examples_per_cluster: int = 50
    max_words: int = 5
    temperature: float = 0.0
    system_prompt: str = (
        "You are a market research analyst. Summarize the theme of the provided survey segments."
    )


@dataclass
class CoderConfig:
    """Controls for mapping cluster labels back to responses."""

    include_examples: bool = True
    output_path: Path = Path("data") / "coded_responses.json"


@dataclass
class GeminiConfig:
    """Configuration for the Gemini LLM backend."""

    model_name: str = "gemini-2.5-flash"
    api_key_env: str = "GOOGLE_API_KEY"
    api_key: Optional[str] = None
    temperature: float = 0.0


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace local models."""

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "auto"
    temperature: float = 0.0
    max_new_tokens: int = 512
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class StorageConfig:
    """File system locations for intermediate artifacts."""

    base_dir: Path = Path("data")
    segments_path: Path = base_dir / "segments.parquet"
    embeddings_path: Path = base_dir / "embeddings.npy"
    topic_hierarchy_path: Path = base_dir / "topic_hierarchy.json"
    nets_path: Path = base_dir / "nets_v1.json"
    assignments_path: Path = base_dir / "assignments.parquet"


@dataclass
class DiscoverySubnetConfig:
    """Controls recursive subnet discovery."""

    enabled: bool = True
    min_size: int = 25
    max_iterations: int = 2
    max_subnets: int = 6
    seeds_per_subnet: int = 6
    assignment_threshold: float = 0.45
    multi_label_delta: float = 0.03
    variance_threshold: float = 0.01


@dataclass
class DiscoveryConfig:
    """LLM-guided recursive net discovery configuration."""

    enabled: bool = True
    max_iterations: int = 3
    assignment_threshold: float = 0.4
    max_bootstrap_nets: int = 12
    bootstrap_rounds: int = 1
    bootstrap_seeds_per_net: int = 7
    min_net_size: int = 20
    min_net_size_pct: Optional[float] = None  # % of total segments (e.g., 0.001 = 0.1%)
    merge_similarity_threshold: float = 0.85
    unknown_label: str = "Unknown"
    max_unknown_clusters: int = 8
    improvement_tol: float = 0.01
    max_llm_calls: Optional[int] = None
    max_refresh_per_iteration: Optional[int] = None
    max_subnet_refresh_per_iteration: Optional[int] = None
    min_parent_child_similarity: float = 0.5
    min_subnet_fraction: float = 0.03
    split_silhouette_threshold: float = 0.01
    max_total_nets: Optional[int] = 300
    duplicate_similarity_threshold: float = 0.85
    dedupe_enabled: bool = True
    # Diversity sampling (percentage-based for scalability)
    bootstrap_sample_pct: Optional[float] = None  # % of total segments (e.g., 0.10 = 10%)
    unknown_sample_pct: Optional[float] = None  # % of unknown segments (e.g., 0.30 = 30%)
    subnet_sample_pct: Optional[float] = None  # % of subnet members (e.g., 0.40 = 40%)
    # Fallback absolute sizes (used if percentages not set)
    bootstrap_sample_size: int = 1000
    unknown_sample_size: int = 800
    subnet_sample_size: int = 500
    refresh_sample_size: int = 200  # For net refresh sampling
    # Advanced/internal parameters (rarely changed)
    multi_label_delta: float = 0.03
    reassign_margin_quantile: float = 0.2
    skip_stability_delta: float = 0.01
    skip_stability_patience: int = 1
    deterministic: bool = True
    synonym_map: Dict[str, str] = field(default_factory=dict)
    subnet: DiscoverySubnetConfig = field(default_factory=DiscoverySubnetConfig)


@dataclass
class PipelineConfig:
    """Aggregated configuration for the entire pipeline."""

    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    labeler: LabelerConfig = field(default_factory=LabelerConfig)
    coder: CoderConfig = field(default_factory=CoderConfig)
    llm_backend: Literal["gemini", "huggingface"] = "gemini"
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    enable_checkpointing: bool = True

    def ensure_storage_paths(self) -> None:
        """Create any directories required by the configured storage paths."""
        self.storage.base_dir.mkdir(parents=True, exist_ok=True)


def pipeline_config_from_dict(payload: Mapping[str, Any]) -> PipelineConfig:
    """Build a PipelineConfig from a nested dictionary."""
    config = PipelineConfig()
    if "llm_backend" in payload:
        config.llm_backend = payload["llm_backend"]
    for field_name in ("segmenter", "embedder", "labeler", "coder", "storage", "gemini", "huggingface"):
        if field_name in payload:
            existing = getattr(config, field_name)
            overrides = dict(payload[field_name])
            if field_name == "storage":
                overrides = {
                    key: Path(value) if isinstance(value, str) else value
                    for key, value in overrides.items()
                }
            if field_name == "coder" and "output_path" in overrides and isinstance(
                overrides["output_path"], str
            ):
                overrides["output_path"] = Path(overrides["output_path"])
            updated = replace(existing, **overrides)
            if field_name == "storage" and "base_dir" in overrides:
                base = updated.base_dir
                defaults = {
                    "segments_path": base / "segments.parquet",
                    "embeddings_path": base / "embeddings.npy",
                    "topic_hierarchy_path": base / "topic_hierarchy.json",
                    "nets_path": base / "nets_v1.json",
                    "assignments_path": base / "assignments.parquet",
                }
                for key, value in defaults.items():
                    if key not in payload[field_name]:
                        updated = replace(updated, **{key: value})
            setattr(config, field_name, updated)
    if "discovery" in payload:
        discovery_overrides = dict(payload["discovery"])
        if "subnet" in discovery_overrides:
            subnet_overrides = dict(discovery_overrides.pop("subnet"))
        else:
            subnet_overrides = {}
        config.discovery = replace(config.discovery, **discovery_overrides)
        if subnet_overrides:
            config.discovery.subnet = replace(config.discovery.subnet, **subnet_overrides)
    if "enable_checkpointing" in payload:
        config.enable_checkpointing = payload["enable_checkpointing"]
    return config


def load_config(path: Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML or JSON file."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configuration files.")
        with open(path, "r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
    elif suffix == ".json":
        import json

        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    return pipeline_config_from_dict(payload)

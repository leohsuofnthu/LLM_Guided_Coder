"""
Survey Topic Discovery Pipeline package.
"""

from .config import (
    GeminiConfig,
    HuggingFaceConfig,
    PipelineConfig,
    load_config,
    pipeline_config_from_dict,
)
from .llm_client import GeminiLLMClient, LLMClient
from .pipeline import SurveyTopicPipeline, run_pipeline
from .recursive_net import RecursiveNetDiscovery

try:
    from .llm_client_hf import HuggingFaceLLMClient
except ImportError:
    HuggingFaceLLMClient = None

__all__ = [
    "PipelineConfig",
    "GeminiConfig",
    "HuggingFaceConfig",
    "LLMClient",
    "GeminiLLMClient",
    "HuggingFaceLLMClient",
    "SurveyTopicPipeline",
    "RecursiveNetDiscovery",
    "run_pipeline",
    "load_config",
    "pipeline_config_from_dict",
]

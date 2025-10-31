"""Compatibility package for the survey topic pipeline.

The core implementation lives in ``survey_topic_pipeline/src``. This module
re-exports the public API so consumers can simply import from
``survey_topic_pipeline`` without worrying about the internal layout.
"""

from . import src as _src

GeminiConfig = _src.GeminiConfig
HuggingFaceConfig = _src.HuggingFaceConfig
PipelineConfig = _src.PipelineConfig
GeminiLLMClient = _src.GeminiLLMClient
HuggingFaceLLMClient = _src.HuggingFaceLLMClient
LLMClient = _src.LLMClient
RecursiveNetDiscovery = _src.RecursiveNetDiscovery
SurveyTopicPipeline = _src.SurveyTopicPipeline
load_config = _src.load_config
pipeline_config_from_dict = _src.pipeline_config_from_dict
run_pipeline = _src.run_pipeline

__all__ = list(_src.__all__)

# Clean up the temporary alias to avoid leaking it.
del _src

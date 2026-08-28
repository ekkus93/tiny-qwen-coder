"""Programming-language specifications and registry services."""

from tiny_qwen_coder.languages.schema import (
    ConfigReferences,
    LanguageConfig,
    LanguageHookReferences,
    RepositoryDetectionSignals,
    SystemPromptSpec,
)
from tiny_qwen_coder.languages.spec import (
    LanguageComponentRef,
    LanguagePlugin,
    LanguageSpec,
    ProtectedBenchmarkRef,
    StaticLanguagePlugin,
)

__all__ = [
    "ConfigReferences",
    "LanguageComponentRef",
    "LanguageConfig",
    "LanguageHookReferences",
    "LanguagePlugin",
    "LanguageSpec",
    "ProtectedBenchmarkRef",
    "RepositoryDetectionSignals",
    "StaticLanguagePlugin",
    "SystemPromptSpec",
]

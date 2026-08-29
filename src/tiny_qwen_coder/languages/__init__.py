"""Programming-language specifications and registry services."""

from tiny_qwen_coder.languages.loading import (
    DEFAULT_EXECUTION_HOOK_ID,
    PRIMARY_VALIDATOR_ID,
    LanguageConfigError,
    load_language_config,
    load_language_plugin,
    parse_language_config,
    plugin_from_language_config,
)
from tiny_qwen_coder.languages.registry import (
    LanguageRegistrationError,
    LanguageRegistry,
    LanguageRegistryError,
    UnknownLanguageError,
)
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
    "DEFAULT_EXECUTION_HOOK_ID",
    "LanguageComponentRef",
    "LanguageConfig",
    "LanguageConfigError",
    "LanguageHookReferences",
    "LanguagePlugin",
    "LanguageRegistrationError",
    "LanguageRegistry",
    "LanguageRegistryError",
    "LanguageSpec",
    "PRIMARY_VALIDATOR_ID",
    "ProtectedBenchmarkRef",
    "RepositoryDetectionSignals",
    "StaticLanguagePlugin",
    "SystemPromptSpec",
    "UnknownLanguageError",
    "load_language_config",
    "load_language_plugin",
    "parse_language_config",
    "plugin_from_language_config",
]

"""Strict loading for declarative programming-language configs and plugins."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from tiny_qwen_coder.languages.schema import (
    ConfigReferences,
    LanguageConfig,
    LanguageHookReferences,
    RepositoryDetectionSignals,
    SystemPromptSpec,
)
from tiny_qwen_coder.languages.spec import (
    LanguageComponentRef,
    LanguageSpec,
    StaticLanguagePlugin,
)

PRIMARY_VALIDATOR_ID = "primary"
DEFAULT_EXECUTION_HOOK_ID = "default"


class LanguageConfigError(ValueError):
    """Raised when a declarative language config is malformed."""


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LanguageConfigError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise LanguageConfigError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise LanguageConfigError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise LanguageConfigError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise LanguageConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LanguageConfigError(f"{context}.{key} must be an integer")
    return value


def _expect_str_tuple(mapping: Mapping[str, object], key: str, *, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list):
        raise LanguageConfigError(f"{context}.{key} must be a YAML sequence")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise LanguageConfigError(f"{context}.{key}[{index}] must be a non-empty string")
        output.append(item)
    return tuple(output)


def _parse_repository_detection(value: object) -> RepositoryDetectionSignals:
    context = "language.repository_detection"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"files", "directories", "globs"}),
        context=context,
    )
    return RepositoryDetectionSignals(
        files=_expect_str_tuple(mapping, "files", context=context),
        directories=_expect_str_tuple(mapping, "directories", context=context),
        globs=_expect_str_tuple(mapping, "globs", context=context),
    )


def _parse_system_prompt(value: object) -> SystemPromptSpec:
    context = "language.system_prompt"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"version", "text"}), context=context)
    return SystemPromptSpec(
        version=_expect_str(mapping, "version", context=context),
        text=_expect_str(mapping, "text", context=context),
    )


def _parse_config_refs(value: object) -> ConfigReferences:
    context = "language.config_refs"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"data_sources", "evaluation"}),
        context=context,
    )
    return ConfigReferences(
        data_sources=_expect_str_tuple(mapping, "data_sources", context=context),
        evaluation=_expect_str_tuple(mapping, "evaluation", context=context),
    )


def _parse_hooks(value: object) -> LanguageHookReferences:
    context = "language.hooks"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"validator", "executor"}), context=context)
    return LanguageHookReferences(
        validator=_expect_str(mapping, "validator", context=context),
        executor=_expect_str(mapping, "executor", context=context),
    )


def parse_language_config(value: object) -> LanguageConfig:
    """Parse one strict declarative language config mapping."""

    context = "language"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "id",
                "aliases",
                "extensions",
                "repository_detection",
                "system_prompt",
                "config_refs",
                "hooks",
            }
        ),
        context=context,
    )
    try:
        return LanguageConfig(
            schema_version=_expect_int(mapping, "schema_version", context=context),
            id=_expect_str(mapping, "id", context=context),
            aliases=_expect_str_tuple(mapping, "aliases", context=context),
            extensions=_expect_str_tuple(mapping, "extensions", context=context),
            repository_detection=_parse_repository_detection(mapping["repository_detection"]),
            system_prompt=_parse_system_prompt(mapping["system_prompt"]),
            config_refs=_parse_config_refs(mapping["config_refs"]),
            hooks=_parse_hooks(mapping["hooks"]),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LanguageConfigError):
            raise
        raise LanguageConfigError(str(exc)) from exc


def load_language_config(path: Path) -> LanguageConfig:
    """Load one strict language config from YAML."""

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LanguageConfigError(f"could not read language config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise LanguageConfigError(f"invalid YAML in {path}: {exc}") from exc
    return parse_language_config(loaded)


def plugin_from_language_config(config: LanguageConfig) -> StaticLanguagePlugin:
    """Build the generic static plugin surface declared by one language config."""

    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(
                id=DEFAULT_EXECUTION_HOOK_ID,
                import_ref=config.hooks.executor,
            ),
            validators=(
                LanguageComponentRef(
                    id=PRIMARY_VALIDATOR_ID,
                    import_ref=config.hooks.validator,
                ),
            ),
        )
    )


def load_language_plugin(path: Path) -> StaticLanguagePlugin:
    """Load a language config and build its generic static plugin."""

    return plugin_from_language_config(load_language_config(path))

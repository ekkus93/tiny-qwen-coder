"""Tests for the language-neutral programming-language config schema."""

import pytest

from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageConfig,
    LanguageHookReferences,
    RepositoryDetectionSignals,
    SystemPromptSpec,
)


def _language_config(
    *,
    language_id: str,
    aliases: tuple[str, ...],
    extensions: tuple[str, ...],
    files: tuple[str, ...],
    directories: tuple[str, ...] = (),
    globs: tuple[str, ...] = (),
) -> LanguageConfig:
    return LanguageConfig(
        schema_version=1,
        id=language_id,
        aliases=aliases,
        extensions=extensions,
        repository_detection=RepositoryDetectionSignals(
            files=files,
            directories=directories,
            globs=globs,
        ),
        system_prompt=SystemPromptSpec(
            version="v1",
            text=f"You are an expert {language_id} software engineer.",
        ),
        config_refs=ConfigReferences(
            data_sources=(f"configs/data/{language_id}.yaml",),
            evaluation=(f"configs/eval/{language_id}.yaml",),
        ),
        hooks=LanguageHookReferences(
            validator=f"tiny_qwen_coder.languages.{language_id}:validate",
            executor=f"tiny_qwen_coder.languages.{language_id}:execute",
        ),
    )


def test_same_schema_represents_python_typescript_and_rust() -> None:
    configs = (
        _language_config(
            language_id="python",
            aliases=("py",),
            extensions=(".py",),
            files=("pyproject.toml", "uv.lock"),
        ),
        _language_config(
            language_id="typescript",
            aliases=("ts",),
            extensions=(".ts", ".tsx"),
            files=("package.json", "tsconfig.json"),
            directories=("node_modules",),
        ),
        _language_config(
            language_id="rust",
            aliases=("rs",),
            extensions=(".rs",),
            files=("Cargo.toml", "Cargo.lock"),
            directories=("src",),
        ),
    )

    assert {config.id for config in configs} == {"python", "typescript", "rust"}
    assert all(type(config) is LanguageConfig for config in configs)
    assert all(config.schema_version == 1 for config in configs)
    assert configs[1].extensions == (".ts", ".tsx")
    assert configs[2].hooks.executor.endswith(":execute")


def test_language_id_and_aliases_are_stable_lookup_tokens() -> None:
    with pytest.raises(ValueError, match="language id"):
        _language_config(
            language_id="TypeScript",
            aliases=("ts",),
            extensions=(".ts",),
            files=("tsconfig.json",),
        )

    with pytest.raises(ValueError, match="canonical language id"):
        _language_config(
            language_id="python",
            aliases=("python",),
            extensions=(".py",),
            files=("pyproject.toml",),
        )


def test_extensions_and_repository_signals_are_validated() -> None:
    with pytest.raises(ValueError, match="file extension"):
        _language_config(
            language_id="python",
            aliases=("py",),
            extensions=("py",),
            files=("pyproject.toml",),
        )

    with pytest.raises(ValueError, match="at least one signal"):
        _language_config(
            language_id="python",
            aliases=("py",),
            extensions=(".py",),
            files=(),
        )


def test_config_and_hook_references_are_explicit() -> None:
    with pytest.raises(ValueError, match="data_sources"):
        ConfigReferences(data_sources=(), evaluation=("configs/eval/python.yaml",))

    with pytest.raises(ValueError, match="package.module:attribute"):
        LanguageHookReferences(validator="python.validate", executor="python:execute")

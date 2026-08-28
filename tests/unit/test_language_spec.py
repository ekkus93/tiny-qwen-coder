"""Tests for the language-neutral runtime plugin contract."""

from __future__ import annotations

import pytest

from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageComponentRef,
    LanguageConfig,
    LanguageHookReferences,
    LanguagePlugin,
    LanguageSpec,
    ProtectedBenchmarkRef,
    RepositoryDetectionSignals,
    StaticLanguagePlugin,
    SystemPromptSpec,
)


def _language_config(
    *,
    language_id: str,
    aliases: tuple[str, ...],
    extensions: tuple[str, ...],
    files: tuple[str, ...],
) -> LanguageConfig:
    return LanguageConfig(
        schema_version=1,
        id=language_id,
        aliases=aliases,
        extensions=extensions,
        repository_detection=RepositoryDetectionSignals(files=files),
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


def _plugin(
    *,
    language_id: str,
    aliases: tuple[str, ...],
    extensions: tuple[str, ...],
    files: tuple[str, ...],
) -> StaticLanguagePlugin:
    config = _language_config(
        language_id=language_id,
        aliases=aliases,
        extensions=extensions,
        files=files,
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(
                id="default",
                import_ref=config.hooks.executor,
            ),
            data_adapters=(
                LanguageComponentRef(
                    id="instruct",
                    import_ref=f"tiny_qwen_coder.languages.{language_id}:load_instruct",
                ),
            ),
            validators=(
                LanguageComponentRef(
                    id="syntax",
                    import_ref=config.hooks.validator,
                ),
            ),
            protected_benchmarks=(ProtectedBenchmarkRef(id="smoke"),),
            evaluation_hooks=(
                LanguageComponentRef(
                    id="default",
                    import_ref=f"tiny_qwen_coder.languages.{language_id}:evaluate",
                ),
            ),
        )
    )


def _register_dummy_plugins(plugins: tuple[LanguagePlugin, ...]) -> dict[str, LanguagePlugin]:
    """Minimal test consumer proving P3-002 can accept one common interface."""

    registry: dict[str, LanguagePlugin] = {}
    for plugin in plugins:
        registry[plugin.spec.id] = plugin
    return registry


def test_python_typescript_and_rust_plugins_share_one_interface() -> None:
    plugins: tuple[LanguagePlugin, ...] = (
        _plugin(
            language_id="python",
            aliases=("py",),
            extensions=(".py",),
            files=("pyproject.toml", "uv.lock"),
        ),
        _plugin(
            language_id="typescript",
            aliases=("ts",),
            extensions=(".ts", ".tsx"),
            files=("package.json", "tsconfig.json"),
        ),
        _plugin(
            language_id="rust",
            aliases=("rs",),
            extensions=(".rs",),
            files=("Cargo.toml", "Cargo.lock"),
        ),
    )

    registry = _register_dummy_plugins(plugins)

    assert set(registry) == {"python", "typescript", "rust"}
    assert all(isinstance(plugin, LanguagePlugin) for plugin in plugins)
    assert registry["typescript"].spec.extensions == (".ts", ".tsx")
    assert registry["rust"].spec.repository_detection.files == ("Cargo.toml", "Cargo.lock")


def test_language_spec_uses_config_as_identity_source_of_truth() -> None:
    plugin = _plugin(
        language_id="python",
        aliases=("py",),
        extensions=(".py",),
        files=("pyproject.toml",),
    )

    assert plugin.spec.id == plugin.spec.config.id == "python"
    assert plugin.spec.aliases == ("py",)
    assert plugin.spec.extensions == (".py",)
    assert plugin.spec.execution_hook.import_ref == plugin.spec.config.hooks.executor
    assert plugin.spec.validators[0].import_ref == plugin.spec.config.hooks.validator
    assert plugin.spec.data_adapters[0].id == "instruct"
    assert plugin.spec.protected_benchmarks[0].id == "smoke"
    assert plugin.spec.evaluation_hooks[0].id == "default"


def test_language_component_reference_is_strict() -> None:
    with pytest.raises(ValueError, match="stable lowercase component ID"):
        LanguageComponentRef(id="PythonLoader", import_ref="package.module:load")

    with pytest.raises(ValueError, match="package.module:attribute"):
        LanguageComponentRef(id="loader", import_ref="package.module.load")


def test_language_spec_rejects_duplicate_component_and_benchmark_ids() -> None:
    config = _language_config(
        language_id="python",
        aliases=("py",),
        extensions=(".py",),
        files=("pyproject.toml",),
    )
    duplicate = LanguageComponentRef(id="same", import_ref="package.module:load")

    with pytest.raises(ValueError, match="duplicate component IDs"):
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=config.hooks.executor),
            data_adapters=(duplicate, duplicate),
        )

    with pytest.raises(ValueError, match="duplicate benchmark IDs"):
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=config.hooks.executor),
            protected_benchmarks=(
                ProtectedBenchmarkRef(id="holdout"),
                ProtectedBenchmarkRef(id="holdout"),
            ),
        )


def test_language_spec_rejects_config_runtime_hook_drift() -> None:
    config = _language_config(
        language_id="python",
        aliases=("py",),
        extensions=(".py",),
        files=("pyproject.toml",),
    )

    with pytest.raises(ValueError, match="execution_hook"):
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref="package.module:execute"),
        )

    with pytest.raises(ValueError, match="primary validator"):
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=config.hooks.executor),
            validators=(LanguageComponentRef(id="syntax", import_ref="package.module:validate"),),
        )


def test_future_capability_collections_may_be_empty_during_staged_implementation() -> None:
    config = _language_config(
        language_id="python",
        aliases=("py",),
        extensions=(".py",),
        files=("pyproject.toml",),
    )
    spec = LanguageSpec(
        config=config,
        execution_hook=LanguageComponentRef(id="default", import_ref=config.hooks.executor),
    )

    assert spec.data_adapters == ()
    assert spec.validators == ()
    assert spec.protected_benchmarks == ()
    assert spec.evaluation_hooks == ()

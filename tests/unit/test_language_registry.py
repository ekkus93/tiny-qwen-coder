"""Tests for deterministic language-plugin registration and lookup."""

import pytest

from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageComponentRef,
    LanguageConfig,
    LanguageHookReferences,
    LanguageRegistrationError,
    LanguageRegistry,
    LanguageSpec,
    RepositoryDetectionSignals,
    StaticLanguagePlugin,
    SystemPromptSpec,
    UnknownLanguageError,
)


def _plugin(
    language_id: str,
    aliases: tuple[str, ...],
    extension: str,
) -> StaticLanguagePlugin:
    validator_ref = f"tiny_qwen_coder.languages.{language_id}:validate"
    executor_ref = f"tiny_qwen_coder.languages.{language_id}:execute"
    config = LanguageConfig(
        schema_version=1,
        id=language_id,
        aliases=aliases,
        extensions=(extension,),
        repository_detection=RepositoryDetectionSignals(files=(f"{language_id}.project",)),
        system_prompt=SystemPromptSpec(version="v1", text=f"Write {language_id} code."),
        config_refs=ConfigReferences(
            data_sources=(f"configs/data/{language_id}.yaml",),
            evaluation=(f"configs/eval/{language_id}.yaml",),
        ),
        hooks=LanguageHookReferences(validator=validator_ref, executor=executor_ref),
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="execute", import_ref=executor_ref),
            validators=(LanguageComponentRef(id="validate", import_ref=validator_ref),),
        )
    )


def _python() -> StaticLanguagePlugin:
    return _plugin("python", ("py",), ".py")


def _typescript() -> StaticLanguagePlugin:
    return _plugin("typescript", ("ts",), ".ts")


def _rust() -> StaticLanguagePlugin:
    return _plugin("rust", ("rs",), ".rs")


def test_registry_resolves_canonical_ids_and_aliases() -> None:
    python = _python()
    typescript = _typescript()
    rust = _rust()
    registry = LanguageRegistry((rust, python, typescript))

    assert registry.resolve("python") is python
    assert registry.resolve("py") is python
    assert registry.resolve("typescript") is typescript
    assert registry.resolve("ts") is typescript
    assert registry.resolve("rust") is rust
    assert registry.resolve("rs") is rust


def test_listing_is_sorted_by_canonical_id_not_registration_order() -> None:
    python = _python()
    typescript = _typescript()
    rust = _rust()
    registry = LanguageRegistry((typescript, rust, python))

    assert registry.list_ids() == ("python", "rust", "typescript")
    assert registry.list_plugins() == (python, rust, typescript)


def test_unknown_language_error_is_clear_and_deterministic() -> None:
    registry = LanguageRegistry((_typescript(), _python()))

    with pytest.raises(UnknownLanguageError) as exc_info:
        registry.resolve("javascript")

    assert str(exc_info.value) == (
        "unknown language 'javascript'; registered languages: python, typescript"
    )


def test_empty_registry_reports_no_available_languages() -> None:
    registry = LanguageRegistry()

    with pytest.raises(UnknownLanguageError) as exc_info:
        registry.resolve("python")

    assert str(exc_info.value) == "unknown language 'python'; registered languages: <none>"


@pytest.mark.parametrize(
    "conflicting_plugin, conflicting_token",
    (
        (_plugin("python", ("python3",), ".py"), "python"),
        (_plugin("py", ("python3",), ".pyx"), "py"),
        (_plugin("rust", ("python",), ".rs"), "python"),
        (_plugin("rust", ("py",), ".rs"), "py"),
    ),
)
def test_registration_rejects_canonical_and_alias_collisions_atomically(
    conflicting_plugin: StaticLanguagePlugin,
    conflicting_token: str,
) -> None:
    python = _python()
    registry = LanguageRegistry((python,))

    with pytest.raises(LanguageRegistrationError) as exc_info:
        registry.register(conflicting_plugin)

    assert str(exc_info.value) == (
        f"language lookup token {conflicting_token!r} is already owned by 'python'; "
        f"cannot register {conflicting_plugin.spec.id!r}"
    )
    assert registry.list_ids() == ("python",)
    assert registry.resolve("python") is python
    assert registry.resolve("py") is python

    for token in (conflicting_plugin.spec.id, *conflicting_plugin.spec.aliases):
        if token not in {"python", "py"}:
            with pytest.raises(UnknownLanguageError):
                registry.resolve(token)


def test_lookup_is_exact_and_does_not_case_fold_or_trim() -> None:
    registry = LanguageRegistry((_python(),))

    for selection in ("Python", " PYTHON ", " py "):
        with pytest.raises(UnknownLanguageError):
            registry.resolve(selection)

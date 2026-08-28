"""Smoke tests for the installable Tiny Qwen Coder package."""

from importlib import import_module

_PACKAGE_MODULES = (
    "tiny_qwen_coder",
    "tiny_qwen_coder.adapters",
    "tiny_qwen_coder.data",
    "tiny_qwen_coder.evaluation",
    "tiny_qwen_coder.languages",
    "tiny_qwen_coder.model",
    "tiny_qwen_coder.reporting",
    "tiny_qwen_coder.runtime",
    "tiny_qwen_coder.training",
    "tiny_qwen_coder.versions",
)


def test_package_modules_import() -> None:
    """Every scaffolded top-level package module should import successfully."""
    for module_name in _PACKAGE_MODULES:
        module = import_module(module_name)
        assert module.__name__ == module_name

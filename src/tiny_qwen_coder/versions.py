"""Dependency version diagnostics for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import platform

_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "pytest",
    "ruff",
    "mypy",
)


def dependency_versions() -> dict[str, str]:
    """Return the Python and project dependency versions visible to this environment."""
    versions = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def main() -> None:
    """Print dependency versions in stable key order."""
    for name, version in dependency_versions().items():
        print(f"{name}={version}")


if __name__ == "__main__":
    main()

"""Deterministic registry for programming-language plugins."""

from __future__ import annotations

from collections.abc import Iterable

from tiny_qwen_coder.languages.spec import LanguagePlugin


class LanguageRegistryError(ValueError):
    """Base class for deterministic language-registry failures."""


class LanguageRegistrationError(LanguageRegistryError):
    """Raised when a plugin cannot be registered without ambiguity."""


class UnknownLanguageError(LanguageRegistryError):
    """Raised when a language selection is not registered."""


class LanguageRegistry:
    """Register language plugins and resolve canonical IDs or aliases.

    Canonical IDs and aliases share one lookup namespace. Registration is
    atomic: every lookup token is checked for conflicts before any state is
    mutated. Listing is sorted by canonical ID so it is independent of
    registration order.
    """

    def __init__(self, plugins: Iterable[LanguagePlugin] = ()) -> None:
        self._plugins_by_id: dict[str, LanguagePlugin] = {}
        self._canonical_id_by_token: dict[str, str] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: LanguagePlugin) -> None:
        """Register one plugin, rejecting ID/alias collisions fail-closed."""

        spec = plugin.spec
        lookup_tokens = (spec.id, *spec.aliases)

        for token in sorted(lookup_tokens):
            existing_id = self._canonical_id_by_token.get(token)
            if existing_id is not None:
                raise LanguageRegistrationError(
                    f"language lookup token {token!r} is already owned by "
                    f"{existing_id!r}; cannot register {spec.id!r}"
                )

        self._plugins_by_id[spec.id] = plugin
        for token in lookup_tokens:
            self._canonical_id_by_token[token] = spec.id

    def resolve(self, selection: str) -> LanguagePlugin:
        """Resolve an exact canonical ID or alias to its registered plugin."""

        canonical_id = self._canonical_id_by_token.get(selection)
        if canonical_id is None:
            available = ", ".join(self.list_ids()) or "<none>"
            raise UnknownLanguageError(
                f"unknown language {selection!r}; registered languages: {available}"
            )
        return self._plugins_by_id[canonical_id]

    def list_ids(self) -> tuple[str, ...]:
        """Return registered canonical language IDs in deterministic order."""

        return tuple(sorted(self._plugins_by_id))

    def list_plugins(self) -> tuple[LanguagePlugin, ...]:
        """Return registered plugins ordered by canonical language ID."""

        return tuple(self._plugins_by_id[language_id] for language_id in self.list_ids())

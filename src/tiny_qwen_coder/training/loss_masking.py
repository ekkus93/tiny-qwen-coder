"""Prove and report assistant/completion-only loss masking for chat SFT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.chat_template_utils import (
    get_training_chat_template,
    has_generation_markers,
    is_chat_template_stop_token_trained,
)

from tiny_qwen_coder.model import InspectionTarget, load_inspection_target

LossMaskStrategy: TypeAlias = Literal["trl_assistant_only", "completion_only_fallback"]
ReportFormat: TypeAlias = Literal["text", "json"]
ChatRole: TypeAlias = Literal["system", "user", "assistant"]

_SCHEMA_VERSION = 1
_DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_IGNORE_INDEX = -100


class LossMaskingError(ValueError):
    """Raised when the project cannot prove a safe loss mask."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Minimal text-only conversational message used by loss-mask validation."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise LossMaskingError("chat message content must not be empty")


@dataclass(frozen=True, slots=True)
class LossMaskSpan:
    """Half-open token span that receives training loss."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise LossMaskingError("loss-mask spans must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class LossMaskToken:
    """One rendered token and whether it receives language-model loss."""

    index: int
    token_id: int
    text: str
    receives_loss: bool

    def __post_init__(self) -> None:
        if self.index < 0:
            raise LossMaskingError("token index must be non-negative")
        if self.token_id < 0:
            raise LossMaskingError("token ID must be non-negative")


@dataclass(frozen=True, slots=True)
class TokenizedLossExample:
    """Token IDs plus a proven boolean loss mask."""

    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    strategy: LossMaskStrategy

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise LossMaskingError("tokenized example must contain at least one token")
        if len(self.input_ids) != len(self.loss_mask):
            raise LossMaskingError("input_ids and loss_mask must have identical lengths")
        if not any(self.loss_mask):
            raise LossMaskingError("loss mask must select at least one token")

    def labels(self, *, ignore_index: int = _IGNORE_INDEX) -> tuple[int, ...]:
        """Return causal-LM labels with non-loss tokens replaced by the ignore index."""

        return tuple(
            token_id if receives_loss else ignore_index
            for token_id, receives_loss in zip(self.input_ids, self.loss_mask, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ChatLossMaskReport:
    """Machine-readable proof of which tokens receive chat-SFT loss."""

    schema_version: int
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_class: str
    trl_version: str
    checkpoint_chat_template_sha256: str
    checkpoint_has_generation_markers: bool
    checkpoint_assistant_token_count: int
    trl_training_template_sha256: str | None
    trl_training_template_has_generation_markers: bool
    trl_assistant_token_count: int
    trl_stop_token_trained: bool
    strategy: LossMaskStrategy
    token_count: int
    loss_token_count: int
    ignored_token_count: int
    loss_spans: tuple[LossMaskSpan, ...]
    loss_text: str
    tokens: tuple[LossMaskToken, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise LossMaskingError(
                f"unsupported loss-mask report schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.token_count != len(self.tokens):
            raise LossMaskingError("token_count must equal the number of token records")
        if self.loss_token_count != sum(token.receives_loss for token in self.tokens):
            raise LossMaskingError("loss_token_count is inconsistent with token records")
        if self.ignored_token_count != self.token_count - self.loss_token_count:
            raise LossMaskingError("ignored_token_count is inconsistent with token records")
        if self.loss_token_count <= 0:
            raise LossMaskingError("report must contain at least one loss-bearing token")


def canonical_probe_messages() -> tuple[ChatMessage, ...]:
    """Return the fixed system/user/assistant probe used by P2-006 validation."""

    return (
        ChatMessage(role="system", content="You are a Python coding assistant."),
        ChatMessage(role="user", content="Return x + 1."),
        ChatMessage(
            role="assistant",
            content="```python\ndef inc(x):\n    return x + 1\n```",
        ),
    )


def _message_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    if not messages:
        raise LossMaskingError("messages must contain at least one conversation turn")
    return [{"role": message.role, "content": message.content} for message in messages]


def _template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _require_template(tokenizer: PreTrainedTokenizerBase) -> str:
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise LossMaskingError("tokenizer does not expose a non-empty chat template")
    return template


def _int_sequence(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise LossMaskingError(f"{field_name} must be an unbatched integer sequence")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise LossMaskingError(f"{field_name}[{index}] must be an integer")
        result.append(item)
    return tuple(result)


def _assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    *,
    chat_template: str,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    encoded = tokenizer.apply_chat_template(
        _message_dicts(messages),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        chat_template=chat_template,
    )
    if not isinstance(encoded, Mapping):
        raise LossMaskingError("chat-template tokenization did not return a mapping")
    input_ids = _int_sequence(encoded.get("input_ids"), field_name="input_ids")
    assistant_masks = _int_sequence(encoded.get("assistant_masks"), field_name="assistant_masks")
    if len(input_ids) != len(assistant_masks):
        raise LossMaskingError("assistant mask length does not match input IDs")
    if any(value not in {0, 1} for value in assistant_masks):
        raise LossMaskingError("assistant mask values must be 0 or 1")
    return input_ids, tuple(bool(value) for value in assistant_masks)


def _tokenize_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    *,
    chat_template: str,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    encoded = tokenizer.apply_chat_template(
        _message_dicts(messages),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        chat_template=chat_template,
    )
    return _int_sequence(encoded, field_name="input_ids")


def completion_only_fallback(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
    *,
    chat_template: str | None = None,
) -> TokenizedLossExample:
    """Mask only the final assistant completion using a proven prefix boundary.

    This fallback is intentionally narrower than assistant-only loss over arbitrary
    multi-turn conversations. It is valid only when the final message is an
    assistant turn and tokenizing the preceding conversation with a generation
    prompt is an exact prefix of tokenizing the completed conversation.
    """

    if len(messages) < 2 or messages[-1].role != "assistant":
        raise LossMaskingError("completion-only fallback requires a final assistant message")
    template = chat_template or _require_template(tokenizer)
    prompt_ids = _tokenize_chat(
        tokenizer,
        messages[:-1],
        chat_template=template,
        add_generation_prompt=True,
    )
    completed_ids = _tokenize_chat(
        tokenizer,
        messages,
        chat_template=template,
        add_generation_prompt=False,
    )
    if len(completed_ids) <= len(prompt_ids):
        raise LossMaskingError("completed conversation does not add any assistant tokens")
    if completed_ids[: len(prompt_ids)] != prompt_ids:
        raise LossMaskingError(
            "completion-only fallback cannot prove a stable prompt/completion token boundary"
        )
    loss_mask = (False,) * len(prompt_ids) + (True,) * (len(completed_ids) - len(prompt_ids))
    return TokenizedLossExample(
        input_ids=completed_ids,
        loss_mask=loss_mask,
        strategy="completion_only_fallback",
    )


def _loss_spans(mask: Sequence[bool]) -> tuple[LossMaskSpan, ...]:
    spans: list[LossMaskSpan] = []
    start: int | None = None
    for index, receives_loss in enumerate((*mask, False)):
        if receives_loss and start is None:
            start = index
        elif not receives_loss and start is not None:
            spans.append(LossMaskSpan(start=start, end=index))
            start = None
    return tuple(spans)


def _decode_one(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    return cast(
        str,
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )


def build_chat_loss_mask_report(
    tokenizer: PreTrainedTokenizerBase,
    target: InspectionTarget,
    messages: Sequence[ChatMessage],
) -> ChatLossMaskReport:
    """Prove assistant-only masking, falling back only to a safe completion boundary."""

    checkpoint_template = _require_template(tokenizer)
    _, checkpoint_mask = _assistant_mask(
        tokenizer,
        messages,
        chat_template=checkpoint_template,
    )
    checkpoint_assistant_count = sum(checkpoint_mask)

    training_template: str | None
    try:
        training_template = get_training_chat_template(tokenizer)
    except (TypeError, ValueError):
        training_template = None
    effective_training_template = training_template or checkpoint_template
    training_has_markers = has_generation_markers(effective_training_template)

    trl_assistant_count = 0
    stop_token_trained = False
    tokenized: TokenizedLossExample | None = None
    if training_has_markers:
        training_ids, training_mask = _assistant_mask(
            tokenizer,
            messages,
            chat_template=effective_training_template,
        )
        trl_assistant_count = sum(training_mask)
        stop_token_trained = is_chat_template_stop_token_trained(
            tokenizer,
            chat_template=effective_training_template,
        )
        if trl_assistant_count > 0 and stop_token_trained:
            tokenized = TokenizedLossExample(
                input_ids=training_ids,
                loss_mask=training_mask,
                strategy="trl_assistant_only",
            )

    if tokenized is None:
        tokenized = completion_only_fallback(
            tokenizer,
            messages,
            chat_template=checkpoint_template,
        )

    selected_ids = [
        token_id
        for token_id, receives_loss in zip(tokenized.input_ids, tokenized.loss_mask, strict=True)
        if receives_loss
    ]
    tokens = tuple(
        LossMaskToken(
            index=index,
            token_id=token_id,
            text=_decode_one(tokenizer, token_id),
            receives_loss=tokenized.loss_mask[index],
        )
        for index, token_id in enumerate(tokenized.input_ids)
    )
    loss_text = cast(
        str,
        tokenizer.decode(
            selected_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )
    return ChatLossMaskReport(
        schema_version=_SCHEMA_VERSION,
        tokenizer_repository=target.tokenizer_repository,
        tokenizer_revision=target.tokenizer_revision,
        tokenizer_class=type(tokenizer).__name__,
        trl_version=importlib.metadata.version("trl"),
        checkpoint_chat_template_sha256=_template_sha256(checkpoint_template),
        checkpoint_has_generation_markers=has_generation_markers(checkpoint_template),
        checkpoint_assistant_token_count=checkpoint_assistant_count,
        trl_training_template_sha256=(
            _template_sha256(effective_training_template) if training_has_markers else None
        ),
        trl_training_template_has_generation_markers=training_has_markers,
        trl_assistant_token_count=trl_assistant_count,
        trl_stop_token_trained=stop_token_trained,
        strategy=tokenized.strategy,
        token_count=len(tokenized.input_ids),
        loss_token_count=sum(tokenized.loss_mask),
        ignored_token_count=len(tokenized.input_ids) - sum(tokenized.loss_mask),
        loss_spans=_loss_spans(tokenized.loss_mask),
        loss_text=loss_text,
        tokens=tokens,
    )


def loss_mask_report_json(report: ChatLossMaskReport) -> str:
    """Serialize a loss-mask proof deterministically as JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def loss_mask_report_text(report: ChatLossMaskReport) -> str:
    """Render a token-level human-readable loss-mask proof."""

    lines = [
        "Tiny Qwen Coder chat loss-mask validation",
        "========================================",
        f"Tokenizer: {report.tokenizer_repository}@{report.tokenizer_revision}",
        f"Tokenizer class: {report.tokenizer_class}",
        f"TRL: {report.trl_version}",
        f"Checkpoint template SHA-256: {report.checkpoint_chat_template_sha256}",
        f"Checkpoint generation markers: {report.checkpoint_has_generation_markers}",
        f"Checkpoint assistant-mask tokens: {report.checkpoint_assistant_token_count}",
        f"TRL training template SHA-256: {report.trl_training_template_sha256}",
        f"TRL training generation markers: {report.trl_training_template_has_generation_markers}",
        f"TRL assistant-mask tokens: {report.trl_assistant_token_count}",
        f"TRL stop token trained: {report.trl_stop_token_trained}",
        f"Selected strategy: {report.strategy}",
        f"Tokens: {report.token_count}; loss={report.loss_token_count}; ignored={report.ignored_token_count}",
        f"Loss spans: {[(span.start, span.end) for span in report.loss_spans]}",
        f"Loss text: {report.loss_text!r}",
        "",
        "index  label   token_id  decoded",
        "-----  ------  --------  -------",
    ]
    for token in report.tokens:
        label = "LOSS" if token.receives_loss else "IGNORE"
        lines.append(f"{token.index:5d}  {label:6s}  {token.token_id:8d}  {token.text!r}")
    return "\n".join(lines) + "\n"


def _write_output(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def validate_loss_masking_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for P2-006 pinned chat-template masking validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=_DEFAULT_BASE_CONFIG)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    target = load_inspection_target(args.base_config)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            target.tokenizer_repository,
            revision=target.tokenizer_revision,
        ),
    )
    report = build_chat_loss_mask_report(tokenizer, target, canonical_probe_messages())
    rendered = (
        loss_mask_report_json(report) if args.format == "json" else loss_mask_report_text(report)
    )
    _write_output(rendered, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(validate_loss_masking_main())

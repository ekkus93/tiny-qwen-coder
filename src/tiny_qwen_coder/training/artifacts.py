"""Completed adapter-manifest construction for generic training runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
from typing import Any, cast

from tiny_qwen_coder.adapters import (
    AdapterBaseIdentity,
    AdapterLoraMetadata,
    AdapterManifest,
    AdapterTokenizerIdentity,
    AdapterTrainingProvenance,
    AdapterTrainingSummary,
    ChatTemplateIdentity,
    DatasetManifestReference,
    TrainingComponentSettings,
    TrainingSetting,
    ValidationMetric,
)
from tiny_qwen_coder.reporting import RunManifest
from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


def _effective_chat_template(plan: AdapterTrainingPlan, tokenizer: object) -> ChatTemplateIdentity:
    checkpoint_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(checkpoint_template, str) or not checkpoint_template:
        raise AdapterTrainingError("training tokenizer does not expose a non-empty chat template")

    template = checkpoint_template
    identifier = "pinned_checkpoint"
    if plan.config.loss_mode == "assistant_only":
        try:
            from trl.chat_template_utils import get_training_chat_template

            candidate = cast(Any, get_training_chat_template)(tokenizer)
        except (ImportError, TypeError, ValueError):
            candidate = None
        if isinstance(candidate, str) and candidate:
            template = candidate
            identifier = "trl_training_template"

    return ChatTemplateIdentity(
        identifier=identifier,
        sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )


def _adapter_target_modules(model: object) -> tuple[str, ...]:
    target = model
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        target = get_base_model()
    named_modules = getattr(target, "named_modules", None)
    if not callable(named_modules):
        raise AdapterTrainingError("trained model does not expose named_modules()")

    targets: list[str] = []
    for name, module in named_modules():
        if name and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            targets.append(str(name))
    resolved = tuple(sorted(set(targets)))
    if not resolved:
        raise AdapterTrainingError("trained model contains no resolved LoRA target modules")
    return resolved


def _trainable_parameter_count(model: object) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise AdapterTrainingError("trained model does not expose parameters()")
    count = sum(parameter.numel() for parameter in parameters() if parameter.requires_grad)
    if count <= 0:
        raise AdapterTrainingError("trained adapter has no trainable parameters")
    return count


def create_completed_adapter_manifest(
    plan: AdapterTrainingPlan,
    run_manifest: RunManifest,
    *,
    tokenizer: object,
    model: object,
    global_steps: int,
    validation_loss: float | None,
    peak_vram_bytes: int,
) -> AdapterManifest:
    """Create the portable adapter manifest from measured post-training state."""

    if run_manifest.run_kind != "training":
        raise AdapterTrainingError("adapter manifest requires a training run manifest")
    if run_manifest.language != plan.language:
        raise AdapterTrainingError("run-manifest language does not match the training plan")

    metrics: tuple[ValidationMetric, ...] = (
        ()
        if validation_loss is None
        else (
            ValidationMetric(
                name="loss",
                value=float(validation_loss),
                split="validation",
                unit=None,
            ),
        )
    )
    return AdapterManifest(
        schema_version=1,
        adapter_id=plan.config.adapter_id,
        family=plan.config.adapter_family,
        language=plan.language,
        created_at_utc=run_manifest.created_at_utc,
        base_model=AdapterBaseIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
        ),
        tokenizer=AdapterTokenizerIdentity(
            repository=plan.target.tokenizer_repository,
            revision=plan.target.tokenizer_revision,
            chat_template=_effective_chat_template(plan, tokenizer),
        ),
        training=AdapterTrainingProvenance(
            run_id=run_manifest.run_id,
            git_sha=run_manifest.git.sha,
            config_sha256=plan.config_sha256,
            seed=plan.config.seed,
            transformers_version=importlib.metadata.version("transformers"),
            peft_version=importlib.metadata.version("peft"),
        ),
        datasets=(
            DatasetManifestReference(
                manifest_id=plan.dataset.manifest_id,
                sha256=plan.dataset.sha256,
            ),
        ),
        lora=AdapterLoraMetadata(
            rank=plan.config.lora.rank,
            alpha=plan.config.lora.alpha,
            dropout=plan.config.lora.dropout,
            bias=plan.config.lora.bias,
            target_strategy=plan.config.lora.target_strategy,
            target_modules=_adapter_target_modules(model),
            trainable_parameters=_trainable_parameter_count(model),
        ),
        training_summary=AdapterTrainingSummary(
            precision=plan.config.compute_dtype,
            sequence_length=plan.config.sequence_length,
            optimizer=TrainingComponentSettings(
                name="adamw_torch",
                settings=(
                    TrainingSetting(name="learning_rate", value=plan.config.learning_rate),
                    TrainingSetting(
                        name="micro_batch_size",
                        value=plan.config.micro_batch_size,
                    ),
                    TrainingSetting(
                        name="gradient_accumulation_steps",
                        value=plan.config.gradient_accumulation_steps,
                    ),
                ),
            ),
            scheduler=TrainingComponentSettings(
                name=plan.config.scheduler,
                settings=(
                    TrainingSetting(name="warmup_ratio", value=plan.config.warmup_ratio),
                ),
            ),
            steps=global_steps,
            epochs=plan.config.epochs,
            peak_vram_bytes=peak_vram_bytes,
        ),
        validation_metrics=metrics,
        evaluation_artifacts=(),
    )

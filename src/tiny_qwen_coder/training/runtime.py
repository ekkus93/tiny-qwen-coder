"""CUDA runtime for the language-neutral adapter training plan."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import torch

from tiny_qwen_coder.adapters import adapter_manifest_json
from tiny_qwen_coder.data import load_normalized_training_records_jsonl
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.reporting import create_run_manifest, write_run_manifest
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.training.artifacts import create_completed_adapter_manifest
from tiny_qwen_coder.training.plan import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    TrainerArtifactPaths,
    resolve_adapter_training_plan,
    resolved_config_json,
    training_rows,
)
from tiny_qwen_coder.training.preflight import (
    TrainingPreflightReport,
    run_training_preflight,
    training_preflight_json,
)


@dataclass(frozen=True, slots=True)
class AdapterTrainingResult:
    """Stable references to the artifacts emitted by one completed training run."""

    run_id: str
    global_steps: int
    validation_loss: float | None
    peak_vram_bytes: int
    artifacts: TrainerArtifactPaths


@dataclass(frozen=True, slots=True)
class AdapterTrainingRuntimeOptions:
    """Optional bounded runtime controls used by non-promotable smoke runs."""

    output_dir: Path | None = None
    max_steps: int | None = None
    train_sample_limit: int | None = None
    validation_sample_limit: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_steps", self.max_steps),
            ("train_sample_limit", self.train_sample_limit),
            ("validation_sample_limit", self.validation_sample_limit),
        ):
            if value is not None and value <= 0:
                raise AdapterTrainingError(f"runtime option {field_name} must be greater than zero")


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    raise AdapterTrainingError(f"unsupported compute dtype {name!r}")


def _write_metrics(log_history: object, path: Path) -> None:
    if not isinstance(log_history, list):
        raise AdapterTrainingError("trainer log history must be a list")
    lines: list[str] = []
    for item in log_history:
        if not isinstance(item, dict):
            raise AdapterTrainingError("trainer log-history entries must be mappings")
        lines.append(json.dumps(item, sort_keys=True, ensure_ascii=True))
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _validation_loss(metrics: object) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("eval_loss")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _prepare_output(plan: AdapterTrainingPlan, preflight: TrainingPreflightReport) -> None:
    plan.artifacts.output_dir.mkdir(parents=True, exist_ok=False)
    plan.artifacts.checkpoints.mkdir(parents=True, exist_ok=False)
    plan.artifacts.training_config.write_text(resolved_config_json(plan), encoding="utf-8")
    (plan.artifacts.output_dir / "training-preflight.json").write_text(
        training_preflight_json(preflight),
        encoding="utf-8",
    )
    shutil.copyfile(Path(plan.config.dataset_manifest), plan.artifacts.dataset_manifest)


def _bounded_records(
    records: tuple[Any, ...],
    *,
    limit: int | None,
    split_name: str,
) -> tuple[Any, ...]:
    if limit is None:
        return records
    if len(records) < limit:
        raise AdapterTrainingError(
            f"{split_name} contains {len(records)} records; bounded runtime requires {limit}"
        )
    return records[:limit]


def _load_training_runtime(
    plan: AdapterTrainingPlan,
    *,
    options: AdapterTrainingRuntimeOptions | None = None,
) -> tuple[Any, Any]:
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig
        from trl.trainer.sft_config import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError as exc:
        raise AdapterTrainingError(f"training dependency is unavailable: {exc}") from exc

    runtime_options = options or AdapterTrainingRuntimeOptions()
    dtype = _torch_dtype(plan.config.compute_dtype)
    tokenizer_factory = cast(Any, AutoTokenizer)
    tokenizer: Any = tokenizer_factory.from_pretrained(
        plan.target.tokenizer_repository,
        revision=plan.target.tokenizer_revision,
    )

    model_factory = cast(Any, AutoModelForMultimodalLM)
    model_kwargs: dict[str, object] = {
        "revision": plan.target.model_revision,
        "dtype": dtype,
    }
    if plan.config.training_mode == "qlora_4bit":
        quantization = plan.config.quantization
        if quantization is None:
            raise AdapterTrainingError("QLoRA training requires quantization settings")
        bitsandbytes_config_factory = cast(Any, BitsAndBytesConfig)
        model_kwargs["quantization_config"] = bitsandbytes_config_factory(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization.quant_type,
            bnb_4bit_use_double_quant=quantization.double_quant,
            bnb_4bit_compute_dtype=_torch_dtype(quantization.compute_dtype),
        )
        model_kwargs["device_map"] = {"": 0}

    model: Any = model_factory.from_pretrained(plan.target.model_repository, **model_kwargs)
    model.config.use_cache = False
    if plan.config.training_mode == "qlora_4bit":
        prepare = cast(Any, prepare_model_for_kbit_training)
        model = prepare(
            model,
            use_gradient_checkpointing=plan.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    target_modules: str | list[str]
    if plan.config.lora.target_strategy == "all_linear":
        target_modules = "all-linear"
    else:
        target_modules = list(plan.config.lora.target_modules)
    peft_config: Any = LoraConfig(
        r=plan.config.lora.rank,
        lora_alpha=plan.config.lora.alpha,
        lora_dropout=plan.config.lora.dropout,
        bias=plan.config.lora.bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    train_records = _bounded_records(
        load_normalized_training_records_jsonl(
            plan.train_records,
            expected_language=plan.language,
        ),
        limit=runtime_options.train_sample_limit,
        split_name="training split",
    )
    validation_records = _bounded_records(
        load_normalized_training_records_jsonl(
            plan.validation_records,
            expected_language=plan.language,
        ),
        limit=runtime_options.validation_sample_limit,
        split_name="validation split",
    )
    dataset_factory = cast(Any, Dataset)
    train_dataset: Any = dataset_factory.from_list(
        list(training_rows(train_records, loss_mode=plan.config.loss_mode))
    )
    validation_dataset: Any = dataset_factory.from_list(
        list(training_rows(validation_records, loss_mode=plan.config.loss_mode))
    )

    bounded = runtime_options.max_steps is not None
    checkpoint_dir = (
        runtime_options.output_dir / "checkpoints"
        if runtime_options.output_dir is not None
        else plan.artifacts.checkpoints
    )
    sft_config_factory = cast(Any, SFTConfig)
    args: Any = sft_config_factory(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=plan.config.micro_batch_size,
        gradient_accumulation_steps=plan.config.gradient_accumulation_steps,
        num_train_epochs=plan.config.epochs,
        max_steps=runtime_options.max_steps if runtime_options.max_steps is not None else -1,
        learning_rate=plan.config.learning_rate,
        lr_scheduler_type=plan.config.scheduler,
        warmup_ratio=plan.config.warmup_ratio,
        gradient_checkpointing=plan.config.gradient_checkpointing,
        bf16=plan.config.compute_dtype == "bfloat16",
        max_length=plan.config.sequence_length,
        assistant_only_loss=plan.config.loss_mode == "assistant_only",
        completion_only_loss=plan.config.loss_mode == "completion_only",
        eval_strategy="no" if bounded else "epoch",
        save_strategy="steps" if bounded else "epoch",
        save_steps=runtime_options.max_steps if bounded else 500,
        save_total_limit=1 if bounded else None,
        logging_strategy="steps",
        logging_steps=1,
        optim="adamw_torch",
        report_to="none",
        seed=plan.config.seed,
        data_seed=plan.config.seed,
    )
    trainer_factory = cast(Any, SFTTrainer)
    trainer: Any = trainer_factory(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    return trainer, tokenizer


def run_adapter_training(
    config_path: Path,
    *,
    repo_root: Path = Path("."),
) -> AdapterTrainingResult:
    """Execute one generic LoRA/QLoRA training run and freeze its artifacts."""

    plan = resolve_adapter_training_plan(config_path)
    preflight = run_training_preflight(plan, repo_root=repo_root)

    seed_everything(plan.config.seed)
    _prepare_output(plan, preflight)
    run_manifest = create_run_manifest(
        run_kind="training",
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
        language=plan.language,
        adapter=AdapterIdentity(
            family=plan.config.adapter_family,
            adapter_id=plan.config.adapter_id,
        ),
        seed=plan.config.seed,
        repo_root=repo_root,
    )
    write_run_manifest(run_manifest, plan.artifacts.output_dir)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    trainer, tokenizer = _load_training_runtime(plan)
    trainer.train()
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(str(plan.artifacts.adapter))

    global_steps_value = getattr(trainer.state, "global_step", None)
    if isinstance(global_steps_value, bool) or not isinstance(global_steps_value, int):
        raise AdapterTrainingError("trainer did not report an integer global step count")
    peak_vram_bytes = int(torch.cuda.max_memory_reserved(0))
    validation_loss = _validation_loss(evaluation_metrics)
    _write_metrics(getattr(trainer.state, "log_history", None), plan.artifacts.training_metrics)

    adapter_manifest = create_completed_adapter_manifest(
        plan,
        run_manifest,
        tokenizer=tokenizer,
        model=trainer.model,
        global_steps=global_steps_value,
        validation_loss=validation_loss,
        peak_vram_bytes=peak_vram_bytes,
    )
    plan.artifacts.adapter_manifest.write_text(
        adapter_manifest_json(adapter_manifest),
        encoding="utf-8",
    )
    return AdapterTrainingResult(
        run_id=run_manifest.run_id,
        global_steps=global_steps_value,
        validation_loss=validation_loss,
        peak_vram_bytes=peak_vram_bytes,
        artifacts=plan.artifacts,
    )


def train_adapter(argv: list[str] | None = None) -> NoReturn:
    """CLI entry point for generic language-adapter training."""

    parser = argparse.ArgumentParser(description="Train a generic Tiny Qwen Coder adapter")
    parser.add_argument("--config", type=Path, required=True, help="LoRA training YAML config")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Git repository root used for run-manifest provenance",
    )
    args = parser.parse_args(argv)
    result = run_adapter_training(args.config, repo_root=args.repo_root)
    print(result.artifacts.adapter_manifest)
    raise SystemExit(0)

#!/usr/bin/env python3
"""Validate the pinned Qwen chat template's training loss mask."""

from tiny_qwen_coder.training.loss_masking import validate_loss_masking_main

if __name__ == "__main__":
    raise SystemExit(validate_loss_masking_main())

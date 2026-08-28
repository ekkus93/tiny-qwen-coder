#!/usr/bin/env python3
"""Run the canonical 4-bit QLoRA training-memory preflight."""

from __future__ import annotations

from tiny_qwen_coder.training.memory_preflight import training_memory_preflight_main

if __name__ == "__main__":
    training_memory_preflight_main()

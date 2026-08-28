#!/usr/bin/env python3
"""Thin CLI wrapper for the canonical BF16 model smoke test."""

from tiny_qwen_coder.model.smoke_test import smoke_test_model

if __name__ == "__main__":
    smoke_test_model()

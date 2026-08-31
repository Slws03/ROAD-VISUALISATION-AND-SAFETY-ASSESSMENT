"""Detection runners: real imagery in, canonical detection artifacts out.

Runners are the GPU-side half of the pipeline and are deliberately isolated:
they depend on ultralytics and torch, which the CPU fusion layer must not. The
two halves communicate only through the JSONL files written here, so the whole
CPU layer stays runnable, testable and tunable with no model installed.

Install with `uv sync --extra runners`.
"""

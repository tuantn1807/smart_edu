#!/usr/bin/env python3
"""
Automated Scientific Benchmark & Ablation Study Script for PAAF.

Usage:
    python scripts/run_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_all, write_report, format_report, DEFAULT_OUTPUT


def main() -> int:
    print("\n=======================================================")
    print(" KHỞI ĐỘNG SCIENTIFIC BENCHMARK & ABLATION STUDY PAAF ")
    print("=======================================================\n")

    report = evaluate_all()
    write_report(report, DEFAULT_OUTPUT)

    print(format_report(report))
    print(f"\n[THÀNH CÔNG] Đã lưu kết quả benchmark & ablation study đầy đủ vào {DEFAULT_OUTPUT.parent}/")
    print(f" - {DEFAULT_OUTPUT.parent / 'results.json'}")
    print(f" - {DEFAULT_OUTPUT.parent / 'ablation_study.json'}")
    print(f" - {DEFAULT_OUTPUT.parent / 'benchmark_report.json'}")
    print(f" - {DEFAULT_OUTPUT.parent / 'comparison_table.md'}")
    print(f" - {DEFAULT_OUTPUT.parent / 'comparison_table.tex'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

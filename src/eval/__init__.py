"""
Evaluation and benchmark suite for Pedagogical Agentic AI Framework (PAAF).
"""

from src.eval.benchmark_suite import (
    compute_path_coherence,
    compute_zpd_alignment,
    run_ablation_study,
    generate_markdown_table,
    generate_latex_table,
    export_benchmark_artifacts,
)

__all__ = [
    "compute_path_coherence",
    "compute_zpd_alignment",
    "run_ablation_study",
    "generate_markdown_table",
    "generate_latex_table",
    "export_benchmark_artifacts",
]

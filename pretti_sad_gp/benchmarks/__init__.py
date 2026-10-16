"""Benchmark package: the synthetic multi-step benchmarks used in the ESCAPE-36 poster.

Re-exports the benchmark classes and `get_benchmark` from `benchmarks.benchmarks`. The
flowsheet benchmarks (`benchmark_h2_reforming_fs`, `benchmark_bioethanol_fs`) need
optional dependencies and are not imported here; import their modules directly.
"""

from .benchmarks import (
    BenchmarkBase,
    ComplexPharmaFlowsheet,
    ComplexPharmaFlowsheetV2,
    DiscontinuousPolymerization,
    DiscontinuousPolymerizationV2,
    get_benchmark,
)

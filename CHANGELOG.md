# Changelog

All notable changes to VedaRT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-27

### Added
- Initial release of VedaRT
- Adaptive scheduler with thread/process/async/GPU executors
- ParallelIterator with Rayon-like API (map, filter, fold, reduce)
- GPU runtime with CuPy/Numba support
- Telemetry system with Prometheus export
- Deterministic mode for reproducible execution
- Comprehensive test suite (unit, integration, stress)
- Examples and documentation

### Features
- Zero-boilerplate parallel computing
- Automatic executor selection based on workload
- Dynamic worker scaling using Little's Law
- Type-safe API with full mypy strict compliance
- Rich telemetry with <1% overhead
- GPU acceleration with automatic cost-model offload
- Deterministic replay for debugging flaky tests

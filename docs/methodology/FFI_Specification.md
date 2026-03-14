# FFI Specification: Deterministic Parity Between Research and Production

## Requirement

Reproducibility of research results in the production client requires that the **core evaluation logic** — spatial metric computation (Moran's I, LSAP, JFI), map generation, and swap/permutation semantics — be implemented **once** and shared across the Python research environment and the future Rust/Tauri community application. Operational roles (SA for production, NSGA-II for offline generation) do not by themselves resolve **stochastic divergence**: differing floating-point accumulation, PRNG state propagation, and graph-traversal order between Python and Rust will cause the production app to produce different maps than the academic benchmark for the same seed and parameters.

## Architecture

1. **Single implementation.** The core spatial metric evaluation and map-generation logic (including swap neighbourhoods, adjacency, and distance-weighted home values) must live in a **single codebase** compiled to a **C-ABI** or **WebAssembly (WASM)** binary.

2. **FFI bindings.**  
   - **Python (research/HPC):** Call the shared binary via **PyO3** (Rust crate) or ctypes/cffi so that the same floating-point and graph operations run inside the benchmark and analysis scripts.  
   - **Rust/Tauri client:** Link the same core natively or via WASM so that the production "Generate New Map" path executes the identical evaluation and swap logic.

3. **PRNG and floating-point contract.**  
   - **PRNG state propagation** (e.g. seed → sequence of swap decisions) must be specified so that, for a given seed, Python and Rust consume the same sequence of random decisions when running the same algorithm (e.g. SA).  
   - **Floating-point accumulation** in spatial lag $(\mathbf{W}\mathbf{z})_i$, LSAP, and JFI must be specified (e.g. IEEE 754 double, same order of operations) so that metric values are bit-reproducible across environments.

## Status

This document defines a **required future specification**. The current repository contains a Python-only implementation; the Rust production client does not yet exist. Before claiming that empirical data generated in Python is reproducible in the client, the shared core and FFI architecture above must be implemented and validated (e.g. by running the same seed and algorithm in both Python and the client and comparing objective values and final map IDs).

## References

- PyO3: Rust–Python bindings.  
- WebAssembly: portable binary format for the same core to run in browser or Tauri.  
- Methodology §3 (algorithm roles): SA for production, NSGA-II for offline; this spec ensures that "production" and "research" use the same mathematical core.

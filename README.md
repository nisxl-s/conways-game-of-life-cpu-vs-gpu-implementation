# Conway's Game of Life - Massive-Scale Parallelism (CPU vs. CUDA)
### High Performance Computing (HPC) Benchmarking & Profiling - Group 39

[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Architecture](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4%20(Turing)-76B900)](#benchmarking--experimental-environment)
[![Speedup](https://img.shields.io/badge/Speedup-99.9%C3%97%20Kernel%20%7C%2080.1%C3%97%20End--to--End-brightgreen)](#performance-benchmarks--speedup-analysis)
[![Verification](https://img.shields.io/badge/Verification-100%25%20Bit--Identical-success)](#correctness-verification--cross-validation)

> **Coursework**: CCS3032 - High Performance Computing  
> **Institution**: Faculty of Computing, University of Sri Jayewardenepura  
> **Team**: Group 39

---

## Executive Summary

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is an iconic, zero-player two-dimensional cellular automaton governed by strict 8-neighbour Moore rules. Despite its mathematical simplicity, simulating large grids across hundreds of discrete generations is an archetype of **embarrassingly parallel, stencil-based, memory-bound computing**.

In this project, we scale the simulation up to an immense **2048 × 2048 toroidal universe (4,194,304 cells/generation)** and benchmark a tuned, sequential CPU baseline against two distinct parallel GPU architectures in CUDA:
1. **CPU Baseline**: Optimized single-core C with double-buffering, strict pointer aliasing (`restrict`), and loop-aligned micro-optimizations (`-O2 -falign-loops=64`).
2. **CUDA Naive Kernel**: Thread-per-cell parallel decomposition with 2D warp-aligned thread blocks ($16 \times 16$), mapping the grid directly across NVIDIA Streaming Multiprocessors (SMs) with coalesced global memory transactions and high L2 cache locality.
3. **CUDA Shared-Memory Tiled Kernel**: Block-level cooperative halo-exchange staging tiles ($18 \times 18$) directly inside low-latency `__shared__` memory banks to explore stencil cache efficiency.

### Key Headline Findings
* **99.9x Kernel Compute Acceleration**: CUDA kernel execution dropped 100 generations from **1,894.3 ms** down to **18.97 ms** (just **0.190 ms per generation**).
* **80.1x Real-World End-to-End Speedup**: Even when fully factoring in Host-to-Device ($H \to D$) and Device-to-Host ($D \to H$) PCIe bus transfers, the GPU achieves an **80.1x wall-clock speedup (23.64 ms)**.
* **Tiling vs Cache Trade-Off**: On modern GPU architectures (Nvidia Turing Tesla T4), the naive global-memory kernel actually outperformed the shared-memory tiled kernel by **23.5%**, providing deep architectural insight into hardware warp scheduling, L1/L2 data cache throughput, and synchronization overheads.
* **100% Deterministic Bit-Level Correctness**: Cross-validated with identical LCG seeds - every single cell and every saved animation frame between CPU and GPU across all generations matched perfectly (`diff_cells == 0`, `frames 51/51 identical`).

---

## Performance Benchmarks & Speedup Analysis

All tests benchmark **100 generations** on a **2048 × 2048 grid (4.19M cells/step)**, averaged over **5 timed runs** following untimed warmup passes.

### Summary Table

| Implementation | Kernel Time (ms) | Total Wall-Clock (ms) | Time / Generation | Mcells/s Throughput | Kernel Speedup | End-to-End Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (Single-Core C, 2D Arrays)** | - | **1,894.33 ms** | 18.943 ms | 22.1 Mcells/s | **1.0x** (baseline) | **1.0x** (baseline) |
| **CUDA Naive (Global Memory)** | **18.97 ms** | **23.64 ms** | **0.190 ms** | **2,211.0 Mcells/s** | **99.9x** | **80.1x** |
| **CUDA Tiled (Shared Memory)** | **23.44 ms** | **28.15 ms** | **0.234 ms** | **1,789.4 Mcells/s** | **80.8x** | **67.3x** |

```
Execution Time (100 Generations, Log Scale)
──────────────────────────────────────────────────────────────────────────
CPU Baseline         [========================================] 1,894.3 ms
CUDA Naive (Total)   [=] 23.6 ms  (Kernel: 19.0 ms | H2D/D2H: 4.7 ms)
CUDA Tiled (Total)   [==] 28.2 ms (Kernel: 23.4 ms | H2D/D2H: 4.7 ms)
──────────────────────────────────────────────────────────────────────────
```

### Detailed Run-to-Run Variance
* **CPU Runs**: `1826.6 ms`, `1876.7 ms`, `1903.4 ms`, `1939.9 ms`, `1925.0 ms` (Mean: `1894.3 ms`, Spread: `~113 ms` due to OS scheduling and thread preemption).
* **CUDA Naive Runs**: `19.55 ms`, `19.54 ms`, `18.81 ms`, `18.15 ms`, `18.81 ms` (Mean: `18.97 ms`, highly deterministic and noise-free).
* **PCIe Transfer Overhead**: Host-device memory transfers required **4.67 ms**, accounting for approximately **20%** of total GPU execution time. During the generational loop, zero PCIe transfers occur because buffers are ping-ponged directly on device VRAM via pointer swaps.

---

## Architectural Deep-Dive: Why Naive Beat Tiling

In classical GPU literature, staging grid data in `__shared__` memory is the standard pattern for stencil computations. However, on our **Tesla T4 (sm_75 Turing Architecture)**, the naive kernel was **23.5% faster** than the tiled kernel. Below is the architectural analysis:

```
┌───────────────────────────────────────────────┐
│              Naive Kernel Strategy            │
│  Thread (r, c) reads Moore neighbours         │
│  • Coalesced horizontal warp transactions     │
│  • Hardware Turing unified L1/L2 cache hit rate│
│  • Zero __syncthreads() barrier stalls        │
└───────────────────────────────────────────────┘
                       vs
┌───────────────────────────────────────────────┐
│              Tiled Kernel Strategy            │
│  Load (BS+2)x(BS+2) halo into __shared__      │
│  • Cooperative looping load + modulo boundary │
│  • High register pressure + bank conflict risk│
│  • Required __syncthreads() block-wide barrier│
└───────────────────────────────────────────────┘
```

1. **Modern Hardware Cache Hierarchy**: The Turing architecture features a unified, ultra-wide L1 Data Cache / Shared Memory design alongside large L2 caches. Because adjacent threads in a warp access adjacent cells in row-major order, global memory reads are coalesced into compact 32-byte and 128-byte cache line transactions.
2. **Halo Loading Complexity & Modulo Penalty**: Toroidal wrap-around boundaries require modular arithmetic (`% N`). Computing halo coordinates collaboratively inside the thread loop adds arithmetic instruction overhead.
3. **Barrier Synchronization (`__syncthreads()`)**: Shared memory requires synchronization barriers to ensure all halo cells are resident before computation begins, causing warp divergence and pipeline stalls that outweigh the cache latency savings.

---

## The Simulation Mechanics

### Standard Conway B3/S23 Rules
Every cell in the $2048 \times 2048$ toroidal lattice evaluates its 8 Moore neighbours:
* **Underpopulation**: Any live cell with $< 2$ live neighbours dies.
* **Survival**: Any live cell with $2$ or $3$ live neighbours lives on to the next generation.
* **Overpopulation**: Any live cell with $> 3$ live neighbours dies.
* **Reproduction**: Any dead cell with exactly $3$ live neighbours becomes a live cell.

$$\text{State}_{t+1}(r, c) = (N_{r, c} == 3) \lor (\text{State}_{t}(r, c) \land N_{r, c} == 2)$$

Where $N_{r, c}$ is the sum of Moore neighbours evaluated with toroidal wrap-around boundaries:
$$r_{\text{up}} = (r == 0) \ ? \ N - 1 : r - 1, \quad r_{\text{down}} = (r == N - 1) \ ? \ 0 : r + 1$$
$$c_{\text{left}} = (c == 0) \ ? \ N - 1 : c - 1, \quad c_{\text{right}} = (c == N - 1) \ ? \ 0 : c + 1$$

---

## Repository Structure

```
.
├── cpu vs gpu implementation/
│   └── hpc-game-of-life-cpu-vs-gpu-test-4.ipynb    # Complete end-to-end Colab/Jupyter notebook
│                                                   # (Embedded C/CUDA sources, compilation,
│                                                   #  benchmarks, charts, animation generator)
├── presentation/
│   └── Conways-Game-of-Life-CPU-vs-GPU-Presentation-Group-39.pptx  # Full project presentation deck
├── report/
│   └── GameOfLife_HPC_Report_Group_39.pdf           # Comprehensive technical report (LaTeX/PDF)
└── README.md                                       # Project documentation & benchmark overview
```

### Generated Artifacts
When the notebook is executed, it automatically outputs:
* `benchmark_runs.csv`: Full per-run raw execution logs (CPU, Naive, Tiled).
* `comparison.png`: Publication-ready dual-axis execution time charts and copy breakdowns.
* `frame_cpu.pgm` & `frame_cuda.pgm`: Binary PGM image dumps of the 100th generation.
* `life_cpu.gif`, `life_cuda.gif`, `life_cpu_vs_cuda.gif`: High-resolution animations of lattice evolution.
* `results_summary.json`: Complete machine-readable metrics, hardware specs, and verification status.

---

## Benchmarking & Experimental Environment

| Component | Specifications |
| :--- | :--- |
| **Compute Platform** | Google Colaboratory Accelerated Runtime |
| **Host CPU** | Intel(R) Xeon(R) CPU @ 2.00GHz (4 Logical vCPUs) |
| **Target GPU** | NVIDIA Tesla T4 (Turing, Compute Capability `sm_75`, 15,360 MiB VRAM) |
| **NVIDIA Driver** | 580.178.04 |
| **CUDA Toolkit** | Version 12.8 (nvcc V12.8.93) |
| **Host C/C++ Compiler** | GCC 13.3.0 (`gcc -O2 -falign-loops=64`) |
| **GPU Compiler** | NVCC (`nvcc -O2 -arch=sm_75`) |
| **Grid Dimensions** | $2,048 \times 2,048$ Toroidal Grid ($4,194,304$ total cells) |
| **Generations Simulated** | $100$ discrete steps |
| **Initial Lattice State** | Deterministic linear congruential hash (`SEED = 12345u`, ~30% initial live density) |

---

## How to Run

### Option 1: One-Click Execution via Jupyter Notebook
1. Open `cpu vs gpu implementation/hpc-game-of-life-cpu-vs-gpu-test-4.ipynb` in Google Colab or a Jupyter environment with an NVIDIA GPU.
2. In Colab, navigate to **Runtime** -> **Change runtime type** -> Select **T4 GPU**.
3. Select **Runtime** -> **Run all**. The notebook writes `life_cpu.c` and `life_cuda.cu`, compiles both executables, benchmarks all runs, validates correctness, and exports plots and animations.

### Option 2: Manual Terminal Compilation

#### 1. Compile and Run the CPU Version
```bash
# Compile with loop-alignment and level 2 optimizations
gcc -O2 -falign-loops=64 -o life_cpu life_cpu.c

# Run timing benchmark
./life_cpu

# Run and export final grid to PGM
./life_cpu save
```

#### 2. Compile and Run the CUDA Version
```bash
# Compile targeting Tesla T4 (sm_75) or your specific GPU architecture
nvcc -O2 -arch=sm_75 -o life_cuda life_cuda.cu

# Run both Naive and Tiled benchmarks
./life_cuda

# Run and export final grid to PGM
./life_cuda save
```

---

## Correctness Verification & Cross-Validation

To prove that the GPU's $99.9\times$ acceleration does not sacrifice simulation fidelity, the test suite executes rigorous multi-stage verification:

1. **Deterministic PRNG Grid Initialization**: Both implementations use an identical bit-twiddling hash function (`init_cell(r, c)`) seeded with `12345u` to ensure identical initial states across all 4.19M cells.
2. **Global Cell-Sum Checksum**:
   * Initial Live Cells: `1,258,477`
   * Final Live Cells (Gen 100): Exactly `401,781` for both CPU and CUDA implementations.
3. **Memory Bitwise Comparison**: Binary diff between CPU and CUDA memory grids revealed **`0 differing cells`** (`diff_cells == 0`).
4. **Frame-by-Frame Temporal Verification**: 51 animation frame pairs saved at matching intervals were bitwise identical (`51/51 frame pairs passed`).

```
Verification Suite Result:
[x] Final Grids Match:  TRUE  (0 differences across 4,194,304 cells)
[x] Kernel Output Equiv: TRUE  (Naive vs Tiled grids MATCH)
[x] Frame Pair Diffs:    51/51 IDENTICAL
Status: PASSED (100% Correctness Guaranteed)
```

---

## Authors & Acknowledgments

* **Department**: Faculty of Computing, University of Sri Jayewardenepura
* **Course**: CCS3032 - High Performance Computing
* **Group**: Group 39

---

## License

This repository is submitted as academic coursework for the High Performance Computing module at the University of Sri Jayewardenepura. Provided for educational and reference purposes.

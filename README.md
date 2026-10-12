# Conway's Game of Life - Massive-Scale Parallelism (CPU vs. CUDA)
### High Performance Computing (HPC) Benchmarking - Group 39

[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4%20(Turing)-76B900)](#benchmarking--experimental-environment)
[![Speedup](https://img.shields.io/badge/Speedup-99.9%C3%97%20Kernel%20%7C%2080.1%C3%97%20End--to--End-brightgreen)](#performance-benchmarks--speedup-analysis)
[![Verification](https://img.shields.io/badge/Verification-CPU%20%3D%3D%20GPU%20(0%20differing%20cells)-success)](#correctness-verification)

> **Coursework**: CCS3032 - High Performance Computing  
> **Institution**: Faculty of Computing, University of Sri Jayewardenepura  
> **Team**: Group 39

---

## Executive Summary

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a zero-player two-dimensional cellular automaton. The next state of a cell depends on its own current state and the states of its eight neighbours. Simulating a large grid for many generations is a classic example of an **embarrassingly parallel, stencil-based, memory-bound** workload, because every cell can be updated independently within a generation.

In this project we simulate a **2048 × 2048 toroidal grid (4,194,304 cells per generation) for 100 generations** and compare a tuned sequential CPU program with two CUDA kernels:

1. **CPU baseline**: single-threaded C with a true 2D array, double buffering, `restrict`-qualified array parameters, a `noinline` update function, and `-O2 -falign-loops=64`.
2. **CUDA naive kernel**: one thread per cell, 16 × 16 thread blocks, every thread reads its 8 neighbours (and itself) directly from global memory.
3. **CUDA tiled kernel**: the same decomposition, but each block first loads an 18 × 18 tile (16 × 16 cells plus a 1-cell halo) into `__shared__` memory and computes from there.

### Key Findings

* **99.9× kernel speedup**: the naive CUDA kernel runs 100 generations in **18.97 ms** (0.190 ms per generation) against **1,894.3 ms** for the CPU.
* **80.1× end-to-end speedup**: including the copy of the grid to the GPU and back, the naive CUDA program needs **23.64 ms** in total.
* **Tiling did not help on the T4**: the shared-memory kernel was **23.5% slower** than the naive kernel (23.44 ms vs 18.97 ms), giving 80.8× kernel and 67.3× end-to-end speedup.
* **Identical results**: the final CPU and GPU grids are identical (0 differing cells), the two CUDA kernels give identical grids, and 51 of 51 saved animation frame pairs are identical.

---

## Performance Benchmarks & Speedup Analysis

All tests run **100 generations** on a **2048 × 2048 grid**. Each program was run once as an untimed warm-up and then **5 timed runs**, each in a separate process. Tables show means of the 5 runs.

### Summary Table

| Implementation | Kernel Time (ms) | Total Time (ms) | Time / Generation (kernel) | Throughput (Mcells/s) | Kernel Speedup | End-to-End Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (single-threaded C, 2D array)** | - | **1,894.3** | 18.943 ms | 221.4 | 1.0× (baseline) | 1.0× (baseline) |
| **CUDA naive (global memory)** | **18.97** | **23.64** | **0.190 ms** | 22,109 | **99.9×** | **80.1×** |
| **CUDA tiled (shared memory)** | **23.44** | **28.15** | **0.234 ms** | 17,896 | **80.8×** | **67.3×** |

*Total time for CUDA includes copying the grid to the GPU and the result back. Throughput = 4,194,304 cells × 100 generations ÷ time (CPU: total time; CUDA: kernel time).*

```
Execution time for 100 generations (mean of 5 runs)
------------------------------------------------------------------
CPU                  1894.3 ms
CUDA naive (total)     23.6 ms   (kernel 19.0 ms + copies 4.7 ms)
CUDA tiled (total)     28.2 ms   (kernel 23.4 ms + copies 4.7 ms)
------------------------------------------------------------------
```

### Per-Run Results

| Run | CPU total (ms) | Naive kernel (ms) | Naive total (ms) | Tiled kernel (ms) | Tiled total (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1826.6 | 19.547 | 24.291 | 23.933 | 28.649 |
| 2 | 1876.7 | 19.542 | 24.306 | 23.942 | 28.549 |
| 3 | 1903.4 | 18.806 | 23.561 | 23.415 | 28.174 |
| 4 | 1940.0 | 18.154 | 22.654 | 22.949 | 27.769 |
| 5 | 1925.0 | 18.807 | 23.399 | 22.945 | 27.632 |
| **Mean** | **1894.3** | **18.971** | **23.642** | **23.437** | **28.155** |

### Run-to-Run Variation

Variation = slowest run minus fastest run, as a percentage of the mean:

* **CPU**: about 6% (1826.6 to 1940.0 ms). The first run was the fastest; the reason was not investigated.
* **CUDA naive kernel**: about 7% (18.2 to 19.5 ms).
* **CUDA tiled kernel**: about 4% (22.9 to 23.9 ms).

### Copy Overhead and Memory Traffic

* **Host-device copies** (grid to GPU and result back) take about **4.7 ms**, which is about 20% of the naive total and 17% of the tiled total. During the 100 generations no data is copied: the two device buffers are swapped by pointer.
* **Effective bandwidth** (derived, not measured with a profiler): each generation reads and writes 4,194,304 bytes, so about 8.4 MB per generation and 839 MB for 100 generations. At 19.0 ms this is about 44 GB/s, roughly 14% of the T4's 320 GB/s peak memory bandwidth.

---

## Naive vs Tiled: Why Tiling Did Not Help

The tiled kernel is the textbook optimization for stencils, yet on the Tesla T4 it was **23.5% slower** than the naive kernel (the naive kernel needs about 19% less time). The following reasons are plausible but **were not confirmed with a profiler** (for example Nsight Compute):

1. **Caches already serve the overlapping reads.** Neighbouring threads read overlapping cells, and the L1 and L2 caches on Turing absorb most of this reuse. Reads are coalesced because adjacent threads in a row read adjacent bytes (a warp of 32 threads covers two rows of 16 cells, so it reads two short contiguous segments).
2. **The tile has extra loading work.** A block of 256 threads must load 324 cells (256 + 68 halo cells), with a loop and a modulo operation for the toroidal wrap-around.
3. **Synchronization.** `__syncthreads()` is needed after loading the tile, and warps wait at the barrier.

The tile reduces global loads per cell from 9 to about 1.27, but with a 1-byte cell and a 4 MB grid the cache hierarchy appears to make that saving small compared with the added work. On older GPUs with smaller caches tiling is expected to pay off more.

---

## The Simulation Mechanics

### Conway's B3/S23 Rules

Every cell in the 2048 × 2048 toroidal grid looks at its 8 Moore neighbours:

* **Underpopulation**: a live cell with fewer than 2 live neighbours dies.
* **Survival**: a live cell with 2 or 3 live neighbours stays alive.
* **Overpopulation**: a live cell with more than 3 live neighbours dies.
* **Reproduction**: a dead cell with exactly 3 live neighbours becomes alive.

$$\text{State}_{t+1}(r, c) = (N_{r, c} == 3) \lor (\text{State}_{t}(r, c) \land N_{r, c} == 2)$$

where $N_{r, c}$ is the number of live Moore neighbours, with wrap-around at the edges:

$$r_{\text{up}} = (r == 0) \ ? \ N - 1 : r - 1, \quad r_{\text{down}} = (r == N - 1) \ ? \ 0 : r + 1$$
$$c_{\text{left}} = (c == 0) \ ? \ N - 1 : c - 1, \quad c_{\text{right}} = (c == N - 1) \ ? \ 0 : c + 1$$

### Initial State

The start grid is pseudo-random with about 30% live cells. Each cell value comes from an integer hash of its index combined with the fixed constant `SEED = 12345u`. Both programs use the same function, so they start from the identical grid (1,259,237 live cells of 4,194,304).

---

## Repository Structure

```
.
├── cpu vs gpu implementation/
│   └── hpc-game-of-life-cpu-vs-gpu-test-4.ipynb    # Complete notebook
│                                                   # (C/CUDA sources, compilation, benchmarks,
│                                                   #  verification, charts, animations)
├── presentation/
│   └── Conways-Game-of-Life-CPU-vs-GPU-Presentation-Group-39.pptx  # Project presentation
├── report/
│   └── GameOfLife_HPC_Report_Group_39.pdf           # Technical report
└── README.md                                       # Project overview
```

### Generated Artifacts

When the notebook is run it produces:

* `life_cpu.c`, `life_cuda.cu`: the C and CUDA sources written by the notebook.
* `benchmark_runs.csv`: raw per-run timings (CPU, naive, tiled).
* `comparison.png`: charts of execution time and the kernel/copy split.
* `frame_cpu.pgm`, `frame_cuda.pgm`: the final grid (generation 100) of each program.
* `life_cpu.gif`, `life_cuda.gif`: animations of the top-left 256 × 256 corner, 51 frames each, 512 × 512 pixels (each frame enlarged 2×), 120 ms per frame.
* `life_cpu_vs_cuda.gif`: CPU and CUDA side by side, 1032 × 512 pixels (two 512 × 512 panels with an 8-pixel gap).
* `results_summary.json`: machine-readable timings, hardware information and verification result.

---

## Benchmarking & Experimental Environment

| Component | Specification |
| :--- | :--- |
| **Platform** | Kaggle notebook (GPU accelerator) |
| **Host CPU** | Intel(R) Xeon(R) CPU @ 2.00GHz (4 logical CPUs) |
| **GPU** | NVIDIA Tesla T4 (Turing, compute capability 7.5 / `sm_75`, 40 SMs, 15,360 MiB) |
| **NVIDIA driver** | 580.178.04 |
| **CUDA toolkit** | 12.8 (nvcc V12.8.93) |
| **Host C compiler** | GCC 13.3.0, `gcc -O2 -falign-loops=64` |
| **GPU compiler** | `nvcc -O2 -arch=sm_75` |
| **Grid** | 2048 × 2048 toroidal, 4,194,304 cells (1 byte per cell) |
| **Generations** | 100 |
| **Initial state** | integer hash with `SEED = 12345u`, about 30% live cells |

---

## How to Run

### Option 1: Jupyter / Kaggle notebook

1. Open `cpu vs gpu implementation/hpc-game-of-life-cpu-vs-gpu-test-4.ipynb` in Kaggle or any Jupyter environment with an NVIDIA GPU and the CUDA toolkit (`nvcc`) installed.
2. In Kaggle, enable a GPU accelerator (for example **GPU T4**) in the notebook settings.
3. Run all cells. The notebook writes `life_cpu.c` and `life_cuda.cu`, compiles both programs, runs the benchmark, checks correctness, and creates the charts and animations.

### Option 2: Manual compilation

The programs are written by the notebook as `life_cpu.c` and `life_cuda.cu`. Run the notebook cells that write them, or copy the sources from the notebook.

**CPU version**
```bash
gcc -O2 -falign-loops=64 -o life_cpu life_cpu.c

./life_cpu            # timing run
./life_cpu save       # also writes frame_cpu.pgm
./life_cpu frames 2   # saves animation frames (not timed)
```

**CUDA version**
```bash
# use the compute capability of your GPU (75 = Tesla T4)
nvcc -O2 -arch=sm_75 -o life_cuda life_cuda.cu

./life_cuda            # times the naive and tiled kernels
./life_cuda save       # also writes frame_cuda.pgm
./life_cuda frames 2   # saves animation frames (not timed)
```

---

## Correctness Verification

The GPU results were compared with the CPU results in several ways:

1. **Same initial grid**: both programs use the same integer hash with `SEED = 12345u`. The grid has 1,259,237 live cells at generation 0 (19,696 in the top-left 256 × 256 corner).
2. **Live-cell count**: both programs end with exactly **401,781** live cells after 100 generations.
3. **Cell-by-cell comparison**: the final CPU and GPU grids differ in **0 cells**, and both have the SHA-256 prefix `c360510d69799c7d`.
4. **Kernel comparison**: the naive and tiled CUDA kernels give identical grids.
5. **Frame-by-frame comparison**: the top-left 256 × 256 corner was saved every second generation (0 to 100, 51 frames per program), and **51 of 51** frame pairs are identical byte for byte. The live-cell counts in this corner every 10 generations are also the same for both programs.

```
Verification result
[x] Final CPU and GPU grids identical   (0 differing cells)
[x] Naive and tiled kernels identical
[x] Frame pairs identical               51 of 51
Status: PASSED
```

These checks show that the CUDA programs reproduce the CPU program exactly for this seed and grid size.

---

## Authors & Acknowledgments

* **Department**: Faculty of Computing, University of Sri Jayewardenepura
* **Course**: CCS3032 - High Performance Computing
* **Group**: Group 39

---

## License

This repository is submitted as academic coursework for the High Performance Computing module at the University of Sri Jayewardenepura. It is provided for educational and reference purposes.

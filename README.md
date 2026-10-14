SpMV kernels developed for DARE project
=========

### Hardware/Software Requirements
**Hardware (of BSC HCA cluster):**

* `synth-hca`: `ssh synth-hca`
* `bananaf3`: `salloc -p bananaf3 -N 1 -t 01:00:00`

**Software:**

All appropriate libraries have to be loaded with the relevant `module load` commands. This is done automatically for compiling and running the benchmark via [`make.sh`](./src/make.sh) and [`run.sh`](./run.sh).

* For Banana Pi:

```
module load llvm/EPI-development
```

* For RAVE emulation:

```
module load llvm/cross/EPI-development
module load rave/development/EPI
module load sdv_trace/development
```

### Test different formats

To test different formats/implementations, you can use the class template provided [here](./src/spmv_kernels/spmv_kernel_template.cpp). The new class inherits properties from the `Matrix_Format` class defined [here](./src/spmv_kernels/spmv_kernel.h). A conversion function (`csr_to_format`) and an execution function (`spmv`) have to be defined for the new format.

### How to compile

To compile this benchmark, you need to first edit the [`config.sh`](./config.sh) file with the appropriate benchmark configurations that you want (number of threads, desired matrices, spmv kernels ...). Then you run `make.sh` in the `src` directory.

* When trying to run the benchmark on **synth-hca (RAVE emulation)**, you have to set `RAVE_EMULATION=1` in [`config.sh`](./config.sh), as well as `SDV_TRACING=1`. 
* When trying to run the benchmark on **Banana Pi**, you have to set these variables to 0.

### How to run

To run SpMV you simply execute the [`run.sh`](./run.sh) bash script. All the configuration needed is in the [`config.sh`](./config.sh) file, along with documentation comments for each option.

### What matrices to run

There are two options available, configurable in [`config.sh`](./config.sh) through the `USE_ARTIFICIAL_MATRICES` variable.

* When set to 0, the benchmark uses the matrices defined in [`run.sh`](./src/run.sh) as `matrices_validation` variable and stored in [`validation_matrices`](./validation_matrices/). To download matrices, the [`filter.py`](./validation_matrices/download_matrices/filter.py) script can be used.
* When set to 1, artificial sparse matrices are generated through the generator defined [`artificial-matrix-generator`](./artificial-matrix-generator/) and the parameters defined [`matrix_generation_parameters`](./matrix_generation_parameters/).

In RAVE emulation, reading files from disk (validation matrices) is very slow, so the use of artificial matrices, which are generated on-the-fly, is recommended. The parameters for the artificial matrices are the following:

* `rows`: number of rows
* `cols`: number of columns
* `avg_row_size`: average nonzeros per row
* `std_row_size`: standard deviation of nonzeros per row
* `distribution`: distribution of nonzeros per row (currently "normal" and "gamma" distributions are supported)
* `placement`: placement of nonzeros within row (currently "random", "diagonal" and "simple" are supported)
* `avg_bw`: bandwidth of matrix, which confines the column range that a nonzero can be placed at
* `skew_coefficient`: coefficient of skewness, which indicates the imbalance of row length
* `avg_num_neigh`: average number of neighboring nonzeros within row (neighbors : maximum distance 1 for column values)
* `cross_row_sim`: similarity between adjacent rows (based on column position of nonzeros)
* `random_seed`: seed for the random number generator (need to reproduce results on the same artificial matrices across different platforms)

### SpMV Kernels tested

* `csr_naive`: The baseline, non-vectorized implementation that distributes the work by balancing the row workload across threads.
* `csr_vec`: A generalized, cross-architecture row-wise vectorization strategy. It utilizes masks to set the desired vector length, therefore it always reports 256 avg VL, but still is bound by the row length.
* `csr_vector_riscv`: A row-wise vectorization specifically tailored for RISC-V. Its average VL is directly bound by the matrix's average row size.
* `csr_vector_riscv_bulk`: A RISC-V vectorization approach that packs multiple rows together to maximize throughput during the multiplication phase. However, writing the results to the output vector relies on a serialized, per-row vectorized reduction, resulting in an average VL below the 256 maximum.
* `sell`: The Sliced ELLPACK format, configured with a slice size equal to the vector length (Slice=VL). This necessitates a local transposition of the non-zero arrays to ensure contiguous memory access during wide vector loads, achieving an average VL of 256.
* `sell_sorted`: An optimized variant of the Sliced ELLPACK format that locally sorts each thread's rows by size prior to slicing. This minimizes padding overhead while maintaining an average VL of 256.
* `sell_c_s`: The widely recognized SELL-C-σ format, adapted to RVV intrinsics rather than EPI (source [here](https://gitlab.bsc.es/benchmarks/risc-v-benchmarks/-/blob/master/hpc_benchmarks/spmv-ellpack/spmv/libsellcs-spmv/sellcs_mv_kernels_epi.c)). It achieves an average VL of 256.
* `div_rf`: A custom compression format developed by CSLab ([publication](https://dl.acm.org/doi/10.1145/3721145.3725767)) that achieves a dual compression of both matrix indices and values. Operating under a COO-centric logic, it sustains an average VL of 256.

---

### (Developer notes) Changes from default repository ([SpMV-Research](https://github.com/cslab-ntua/SpMV-Research))

* In [`make.sh`](./src/make.sh), added this small snippet so that RAVE emulation enables or disables the use of VEC_LEN=256 in [`csr_vec.cpp`](./src/spmv_kernels/csr_vec.cpp), [`sell.cpp`](./src/spmv_kernels/sell.cpp), [`sell_sorted.cpp`](./src/spmv_kernels/sell_sorted.cpp) [`div_gather_coords.h`](./src/spmv_kernels/DIV/div_gather_coords.h) files. Otherwise default (vec_len_default_f64) is used.

```
if ((RAVE_EMULATION)); then
    CFLAGS+=" -D'RAVE_EMULATION=1'"
fi
```

* In [`run.sh`](./run.sh), added this small snippet so that library loading is performed automatically.

```
if ((RAVE_EMULATION)); then
    export ARCH='rave'
else
    export ARCH="$(uname -m)"
    # export ARCH='riscv64'
fi
if [[ ${ARCH} == riscv64 || ${ARCH} == rave ]]; then
    case "${ARCH}" in
        riscv64)
            module load llvm/EPI-development
            ;;
        rave)
            module load llvm/cross/EPI-development
            module load rave/development/EPI
            module load sdv_trace/development
            ;;
    esac
fi
```

* In [`sell_sorted.cpp`](./src/spmv_kernels/sell_sorted.cpp), fixed a small bug that lead to segmentation fault.

```
long ii_e_last = ii_e;
if (ii_e * VEC_LEN > i_e){
    // ii_e_last -= VEC_LEN; // this is wrong!
    ii_e_last--;
}
```

* In [`bench_spmv.cpp`](./src/bench_spmv.cpp), removed the check of the L3 cache (and all relevant flags in [`config.sh`](./config.sh)) because it caused a seg fault in Banana Pi for a specific format only. Additionally, removed any rapl-related counters.

* In [`csr_riscv.cpp`](./src/spmv_kernels/csr_riscv.cpp) changed `v_idx_row` to `v_idx_col`.

* In [`div_kernels_rf.h`](./src/spmv_kernels/DIV/div_kernels_rf.h) put Line 405 in comments. We need to disable this quicksort call, due to wide Vector Length.

* Several changes had to be made in order to test this on a custom QEMU installation:

  * In [`config.sh`](./config.sh), added variables for QEMU RISC-V emulation: `OCT_QEMU_EMULATION` (toggle), `QEMU_BIN` (emulator binary path), `QEMU_LD_PREFIX` (`/usr/riscv64-linux-gnu`), and `QEMU_CPU` (CPU model and vector length, e.g. `max,vlen=1024` or `max,vlen=16384`).
  * In [`run.sh`](./run.sh), added `ARCH='qemu'` mode and executed binaries through QEMU instead of native / RAVE:
    ```bash
    "$QEMU_BIN" -L "$QEMU_LD_PREFIX" -cpu "$QEMU_CPU" "$prog" "${prog_args[@]}"
    ```
  * In [`make.sh`](./src/make.sh), added the `qemu` target architecture:
    * Selects GNU cross-compilers (`riscv64-linux-gnu-gcc`, `riscv64-linux-gnu-g++`, `riscv64-linux-gnu-ar`).
    * Sets `-D'OCT_QEMU_EMULATION=1'`.
    * Replaces Clang/EPI-specific flags (`-mcpu=avispado`, `-mepi`, etc.) with GCC RVV 1.0 flags: `-march=rv64gcv -mabi=lp64d -fno-lto -flax-vector-conversions -fno-tree-slp-vectorize`.
    * Adds `-L /usr/riscv64-linux-gnu/lib` to `LDFLAGS`.
  * In [`csr_vec.cpp`](./src/spmv_kernels/csr_vec.cpp), [`sell.cpp`](./src/spmv_kernels/sell.cpp), [`sell_sorted.cpp`](./src/spmv_kernels/sell_sorted.cpp), and [`div_gather_coords.h`](./src/spmv_kernels/DIV/div_gather_coords.h), grouped `OCT_QEMU_EMULATION` with `RAVE_EMULATION` to select wide vector lengths (`VEC_LEN=256` for double, `512` for float).
  * In the `sell-C-s` build system:
    * In [`Makefile_in`](./src/Makefile_in), forwarded `ARCH`, `DOUBLE`, `CC`, and `AR` to the sub-make instead of hardcoding `CC=clang LLVM=clang`.
    * In [`sell-C-s/Makefile`](./src/spmv_kernels/sell-C-s/Makefile), made compiler and archiver overridable (`CC ?= clang`, `LLVM ?= $(CC)`).
    * In [`sell-C-s/setup/riscv.mk`](./src/spmv_kernels/sell-C-s/setup/riscv.mk), added an `ifeq ($(ARCH),qemu)` configuration targeting `riscv64-linux-gnu-gcc` and standard RVV flags without EPI/Clang-only attributes.

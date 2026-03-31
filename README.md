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

To test different formats/implementations, you can use the class template provided [here](./src/spmv_kernel_template.cpp). The new class inherits properties from the `Matrix_Format` class defined [here](./src/spmv_kernel.h). A conversion function (`csr_to_format`) and an execution function (`spmv`) have to be defined for the new format.

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

* `csr_naive`: no vectorization, baseline implementation that balances row-workload among threads.
* `csr_vector_riscv`: row-based vectorization, achieves average vector length (avg VL) equal to the avg row size.
* `csr_vector_riscv_bulk`: vvrettos's vectorization, close to 256 avg VL, but not quite!
* `csr_vec`: generic (across architectures) vectorization , achieves 256 avg VL.
* `sell`: Sliced ELLPACK format, achieves 256 avg VL
* `sell_sorted`: Sliced ELLPACK format, sort rows by size, reduces padding overhead, achieves 256 avg VL
* `sell_c_s`: famous SELL-C-σ format, achieves 256 avg VL, adapted from [here](https://gitlab.bsc.es/benchmarks/risc-v-benchmarks/-/blob/master/hpc_benchmarks/spmv-ellpack/spmv/libsellcs-spmv/sellcs_mv_kernels_epi.c) to work with RVV intrinsics instead of EPI

* **`div_rf`: remains to be tested.**

---

### (Developer notes) Changes from default repository ([SpMV-Research](https://github.com/cslab-ntua/SpMV-Research))

* In [`make.sh`](./src/make.sh), added this small snippet so that RAVE emulation enables or disables the use of VEC_LEN=256 in [`csr_vec.cpp`](./src/spmv_kernels/csr_vec.cpp), [`sell.cpp`](./src/spmv_kernels/sell.cpp), [`sell_sorted.cpp`](./src/spmv_kernels/sell_sorted.cpp) files. Otherwise default (vec_len_default_f64) is used.

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
    ii_e_last -= 1;
}
```

* In [`bench_spmv.cpp`](./src/bench_spmv.cpp), removed the check of the L3 cache (and all relevant flags in [`config.sh`](./config.sh)) because it caused a seg fault in Banana Pi for a specific format only. Additionally, removed any rapl-related counters.


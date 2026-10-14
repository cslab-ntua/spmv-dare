#!/bin/bash

script_dir="$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")"
source "$script_dir"/config.sh
echo

if ((OCT_QEMU_EMULATION)); then
    export ARCH='qemu'
elif ((RAVE_EMULATION)); then
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

# Encourages idle threads to spin rather than sleep.
# export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'

matrices_validation=(
    scircuit
)
validation_dirs=(
    "${path_validation}"
)
matrices_validation=( 
    $(
        for ((i=0;i<${#matrices_validation[@]};i++)); do
            m="${matrices_validation[i]}.mtx"
            for d in "${path_validation[@]}"; do
                if [[ -f "${d}/${m}" ]]; then
                    echo "${d}/${m}"
                    break
                fi
            done
        done
    )
)

bench()
{
    declare args=("$@")
    declare prog="${args[0]}"
    declare prog_args=("${args[@]:1}")
    declare array_cores=( $cores )
    declare array_affinities=()
    declare c
    declare i

    mapfile -t -d ':' array_affinities < <(printf "$cpu_affinity")
    for ((i=0;i<${#array_cores[@]};i++)); do
        c="${array_cores[i]}"
        aff="${array_affinities[i]}"

        export OMP_NUM_THREADS="$c"

        # GOMP_CPU_AFFINITY pins the threads to specific cpus, even when assigning more cores than threads.
        # e.g. with 'GOMP_CPU_AFFINITY=0,1,2,3' and 2 threads, the threads are pinned: t0->core0 and t1->core1.
        export GOMP_CPU_AFFINITY="$aff"

        while :; do
            # If a GPU kernel is tested, then set env variable accordingly. This is used in spmv_bench, so that the warm up is performed 1000 times, instead of just once (CPU warm up)
            export GPU_KERNEL=0

            echo '------------------------'
            prog_name="$(basename "$prog")"
            export PROGG="${prog_name}"
            
            if ((OCT_QEMU_EMULATION)); then
                "$QEMU_BIN" -L "$QEMU_LD_PREFIX" -cpu "$QEMU_CPU" "$prog" "${prog_args[@]}" 2>'tmp.err'
            elif ((RAVE_EMULATION)); then
                trace_rave_1_0 "$prog" "${prog_args[@]}"  2>"tmp.err" # this is located in /apps/riscv/sdv_trace/util
            else
                "$prog" "${prog_args[@]}"  2>'tmp.err'
            fi

            ret="$?"
            if ((output_to_files)); then   # If outputing to files, also print stderr to stdout.
                cat 'tmp.err'
            fi
            if ((!ret || !force_retry_on_error)); then      # If not retrying then print the error text to be able to notice it.
                cat 'tmp.err' >&2
                break
            fi
            echo "ERROR: Program exited with error [${ret}], retrying."
        done
    done

    rm 'tmp.err'
}

if ((!USE_ARTIFICIAL_MATRICES)); then
    prog_args=("${matrices_validation[@]}")
else
    prog_args=()
    tmp=()
    for f in "${artificial_matrices_files[@]}"; do
        IFS=$'\n' read -d '' -a tmp < "$f"
        prog_args+=("${tmp[@]}")
    done
fi

for format_name in "${!progs[@]}"; do
    prog="${progs["$format_name"]}"

    if ((output_to_files)); then
    	mkdir -p out_logs
        > out_logs/"${format_name}.out"
        exec 1>>out_logs/"${format_name}.out"
        > out_logs/"${format_name}.csv"
        exec 2>>out_logs/"${format_name}.csv"
    fi

    echo "$config_str"

    echo "program: $prog"
    echo "number of matrices: ${#prog_args[@]}"

    # Just print the output labels first.
    if ((OCT_QEMU_EMULATION)); then
        "$QEMU_BIN" -L "$QEMU_LD_PREFIX" -cpu "$QEMU_CPU" "$prog"
    elif ((RAVE_EMULATION)); then
        trace_rave_1_0 "$prog"
    else
        "$prog"
    fi

    rep=1
    # rep=5

    ts1="$(date '+%s')"

    for ((i=0;i<rep;i++)); do
        for a in "${prog_args[@]}"; do
            echo "File: $a"
            bench "$prog" $a
        done
    done

    ts2="$(date '+%s')"
    printf "\n\nTotal time = $(( (ts2 - ts1) / 60 )) minutes $(( (ts2 - ts1) % 60 )) seconds\n"

done

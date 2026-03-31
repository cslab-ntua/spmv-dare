#!/bin/bash

script_dir="$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")"

find_valid_dir()
{
    local args=("$@")
    for p in "${args[@]}"; do
        if [[ -d "$p" ]]; then
            printf "$p"
            break
        fi
    done
}

calc_cpu_pinning()
{
    local cores="$1" max_cores="$2" cpu_pinning_step="$3" cpu_pinning_group_size="$4"
    local affinity thread_groups cycle_len i j
    affinity=''
    thread_groups="$(( cores / cpu_pinning_group_size ))"
    for ((i=0;i<thread_groups;i++)); do
        cycle_len="$(( max_cores / cpu_pinning_step ))"
        pos="$(( (i % cycle_len) * cpu_pinning_step + (i / cycle_len) % cpu_pinning_step ))"
        for ((j=0;j<cpu_pinning_group_size;j++)); do
            affinity="$affinity,$((pos + j))"
        done
    done
    affinity="${affinity#,}"
    printf "$affinity"
}

calc_numa_nodes()
{
    local cores="$1" max_cores="$2" affinity="$3"
    local -A dict
    local -a aff
    local -A nodes
    local -A nodes
    local i j c
    numactl_output="$(numactl -H)"
    avail="$(printf "$numactl_output" | awk '/available/ {printf("%d", $2)}')"
    dict=()
    for ((i=0;i<avail;i++)); do
        node_cores=( $(printf "$numactl_output" | awk '/node '"$i"' cpus:/ { for (i=4;i<=NF;i++) {printf("%s ", $i)} }') )   # $i in awk expands the 'i' awk variable and if a number prints the ith line token, else $0.
        for n in "${node_cores[@]}"; do
            dict["$n"]="$i"
        done
    done
    aff=(${affinity//,/' '})
    nodes=()
    for c in "${aff[@]}"; do
        if [[ -v dict["$c"] ]]; then
            nodes["${dict["$c"]}"]=1
        fi
    done
    nodes="$(printf "%s\n" "${!nodes[@]}" | sort -n | tr '\n' ',')"
    nodes="${nodes%,}"
    printf "${nodes}"
}


cpu_info="$(lscpu)"
num_cpus="$(echo "$cpu_info" | awk '/^CPU\(s\):/ {printf("%d", $2);}')"
num_cores="$(( num_cpus / $(echo "$cpu_info" | awk '/^Thread\(s\) per core:/ {printf("%d", $4);}') ))"

declare -A conf_vars
conf_vars=(

    # Whether the RAVE emulator is used.
    # ['RAVE_EMULATION']=0
    ['RAVE_EMULATION']=1

    # Whether to enable tracing with SDV.
    # ['SDV_TRACING']=0
    ['SDV_TRACING']=1

    # ['USE_ARTIFICIAL_MATRICES']=0
    ['USE_ARTIFICIAL_MATRICES']=1

    ['output_to_files']=0
    # ['output_to_files']=1

    # Cores / Threads to utilize. Use spaces to define a set of different thread numbers to benchmark.
    # ['cores']="$num_cores" # always run with maximum number of cores on Banana Pi
    ['cores']=1 # always run with 1 core in synth-hca (RAVE emulation)

    ################################## 

    # ['PRINT_STATISTICS']=0
    ['PRINT_STATISTICS']=1

    ['USE_PROCESSES']=0
    # ['USE_PROCESSES']=1

    ['force_retry_on_error']=0
    # ['force_retry_on_error']=1

    ['COOLDOWN']=0
    # ['COOLDOWN']=1

    # Whether to use RCM reordering on the matrix. Valid only for symmetric matrices.
    ['USE_RCM_REORDERING']=0
    # ['USE_RCM_REORDERING']=1

    # Whether to clear cpu caches before each spmv iteration.
    ['CLEAR_CACHES']=0
    # ['CLEAR_CACHES']=1

    # Value compression error tolerance.
    ['DIV_VC_TOLERANCE']=0
    # ['DIV_VC_TOLERANCE']='1e-12'
    # ['DIV_VC_TOLERANCE']='1e-9'
    # ['DIV_VC_TOLERANCE']='1e-7'
    # ['DIV_VC_TOLERANCE']='1e-6'
    # ['DIV_VC_TOLERANCE']='1e-3'

    # Whether to enable work stealing for the DIV formats. Ignored for the symmetric variant.
    # ['DIV_STEALING_ENABLED']=0
    ['DIV_STEALING_ENABLED']=1

    # Whether to enable packet reordering optimization.
    ['DIV_PACKET_REORDERING']=0
    # ['DIV_PACKET_REORDERING']=1

    # Maximum number of the machine's cores.
    ['max_cores']=256

    # Cpu pinning distance for contiguous thread ids, 1 means adjacent core numbers.
    ['cpu_pinning_step']=1

    # Group size of threads pinned adjacently (e.g. for when hyperthreaded cores have contiguous ids).
    ['cpu_pinning_group_size']=1

    # Thread pinning policy (auto-generated from the above).
    ['cpu_affinity']=''

    # Numa nodes of the selected cores.
    ['numa_nodes']=''

    # Path for the validation matrices.
    ['path_validation']="$( options=(
                        "${script_dir}/validation_matrices"
                    )
                    find_valid_dir "${options[@]}"
                )"
)

cpu_affinity=''
for c in ${conf_vars['cores']}; do
    cpu_affinity="${cpu_affinity}:$(calc_cpu_pinning "$c" "${conf_vars["max_cores"]}" "${conf_vars["cpu_pinning_step"]}" "${conf_vars["cpu_pinning_group_size"]}")"
done
conf_vars['cpu_affinity']="${cpu_affinity:1}"

if hash numactl 2>/dev/null; then
    conf_vars['numa_nodes']="$(calc_numa_nodes "${conf_vars["cores"]}" "${conf_vars["max_cores"]}" "${conf_vars["cpu_affinity"]}")"
else
    conf_vars['numa_nodes']="0"
fi

path_artificial="${script_dir}/matrix_generation_parameters"

# Artificial matrices to benchmark.
artificial_matrices_files=(
    "$path_artificial"/test.txt
)

declare -A progs

# SpMV kernels to benchmark (uncomment the ones you want).
progs=(
    # ['csr_naive_d']="${script_dir}/src/spmv_csr_naive_d.exe"
    # ['csr_vec_d']="${script_dir}/src/spmv_csr_vec_d.exe"

    # ['csr_vector_riscv_d']="${script_dir}/src/spmv_csr_vector_riscv_d.exe"
    ['csr_vector_riscv_bulk_d']="${script_dir}/src/spmv_csr_vector_riscv_bulk_d.exe"

    # ['sell_d']="${script_dir}/src/spmv_sell_d.exe"
    # ['sell_sorted_d']="${script_dir}/src/spmv_sell_sorted_d.exe"

    # ['sell_c_s_d']="${script_dir}/src/spmv_sell_c_s_d.exe"

    # ['div_rf_d']="${script_dir}/src/spmv_div_rf_d.exe"
)

# Export variables for make.
config_str=''
for index in "${!conf_vars[@]}"; do
    eval "export $index='${conf_vars["$index"]}'"
    config_str="${config_str}${index}=${conf_vars["$index"]};"
    # printf "%s=%s;" "$index"  "${conf_vars["$index"]}"
done
printf "%s\n" "$config_str"


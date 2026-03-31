#!/bin/bash

shopt -s -o pipefail
shopt -s extglob globstar dotglob nullglob 2>/dev/null
export GLOBIGNORE=.:..


export CPATH=''
export library=../lib


script_dir="$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")"
source "$script_dir"/../config.sh


export AMG_PATH=../artificial-matrix-generator


if ((RAVE_EMULATION)); then
    export ARCH='rave'
else
    export ARCH="$(uname -m)"
    # export ARCH='riscv64'
fi

gcc_bin=gcc
gpp_bin=g++

if [[ ${ARCH} == rave ]] || [[ ${ARCH} == riscv64 ]]; then
    CC=clang
    CPP=clang++
else
    CC="$gcc_bin"
    CPP="$gpp_bin"
    CXX="$CPP"
fi
export CC
export CPP
export CXX

CFLAGS=''
CFLAGS+=" -Wall -Wextra"
CFLAGS+=" -pipe"  # Tells the compiler to use pipes instead of temporary files (faster compilation, but uses more memory).
# CFLAGS+=" -Wno-unused-variable"
# CFLAGS+=" -Wno-alloc-size-larger-than"
CFLAGS+=" -fsigned-char"
CFLAGS+=" -fopenmp"

CFLAGS+=" -D'_GNU_SOURCE'"

if ((RAVE_EMULATION)); then
    CFLAGS+=" -D'RAVE_EMULATION=1'"
fi

# CFLAGS+=" -g"
# CFLAGS+=" -g3 -fno-omit-frame-pointer"
# CFLAGS+=" -Og"
# CFLAGS+=" -O0"
# CFLAGS+=" -O2"
CFLAGS+=" -O3"

# CFLAGS+=" -ffast-math"

CFLAGS+=" -flto=auto"

CFLAGS+=" -fsigned-char"

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

    CFLAGS+=" -fno-lto"
    CFLAGS+=" -fmacro-backtrace-limit=0"

    CFLAGS+=" -flax-vector-conversions"
    CFLAGS+=" -Wno-vla-cxx-extension"

    # atrevido-vec gets stuck on synth-hca for sell-c-s only...
    CFLAGS+=" -mcpu=avispado"
    # CFLAGS+=" -mcpu=atrevido-vec"

    CFLAGS+=" -mepi"
    CFLAGS+=" -fno-slp-vectorize"
    CFLAGS+=" -mllvm -combiner-store-merging=0"
    CFLAGS+=" -mllvm -vectorizer-use-vp-strided-load-store"
    CFLAGS+=" -mllvm -disable-loop-idiom-memcpy"
    CFLAGS+=" -mllvm -disable-loop-idiom-memset"
    CFLAGS+=" -mllvm -riscv-uleb128-reloc=0"
    CFLAGS+=" -Xclang -target-feature -Xclang +does-not-implement-vszext"
    CFLAGS+=" -Xclang -target-feature -Xclang +does-not-implement-tu"
    # CFLAGS+=" -Rpass=loop-vectorize"
    # CFLAGS+=" -Rpass-analysis=loop-vectorize"
    # CFLAGS+=" -Rpass-missed=loop-vectorize"
else
    CFLAGS+=" -mcpu=native"
fi

CFLAGS+=" -I'${library}'"
CFLAGS+=" -I'${AMG_PATH}'"

CFLAGS+=" -D'INT_T=int32_t' -D'UINT_T=uint32_t'"

if ((${PRINT_STATISTICS} == 1)); then
    CFLAGS+=" -D'PRINT_STATISTICS'"
fi

# Read the matrix in double-precision for checking accuracy against doubles.
CFLAGS+=" -D'ValueTypeReference=double'"
CFLAGS+=" -D'MATRIX_MARKET_FLOAT_T=ValueTypeReference'"

# For synth-hca system only. Enable SDV tracing through RAVE emulation.
if ((RAVE_EMULATION)) && ((SDV_TRACING)); then
    CFLAGS+=" -D'SDV_TRACING' ${SDV_TRACE_INCL}"
fi

CPPFLAGS=''
CPPFLAGS+=" ${CFLAGS}"

CFLAGS_D="${CFLAGS} -D'DOUBLE=1' -D'ValueType=double' -D'ValueTypeI=uint64_t'"
CPPFLAGS_D="${CPPFLAGS} -D'DOUBLE=1' -D'ValueType=double' -D'ValueTypeI=uint64_t'"

CFLAGS_F="${CFLAGS} -D'DOUBLE=0' -D'ValueType=float' -D'ValueTypeI=uint32_t'"
CPPFLAGS_F="${CPPFLAGS} -D'DOUBLE=0' -D'ValueType=float' -D'ValueTypeI=uint32_t'"

LDFLAGS=''
LDFLAGS+=' -lm'

if [[ ${ARCH} == riscv64 ]]; then
    :
    # LDFLAGS+=' -L /usr/riscv64-linux-gnu'
    # LDFLAGS+=' -Wl,-rpath,/usr/riscv64-linux-gnu/usr/lib'
    # LDFLAGS+=' -Wl,-dynamic-linker,/usr/riscv64-linux-gnu/usr/lib/ld-linux-riscv64-lp64d.so.1'
    LDFLAGS+=" -v"
elif [[ ${ARCH} == rave ]]; then
    :
    # LDFLAGS+=' -L /usr/riscv64-linux-gnu'
    # LDFLAGS+=' -Wl,-rpath,/usr/riscv64-linux-gnu/usr/lib'
    # LDFLAGS+=' -Wl,-dynamic-linker,/usr/riscv64-linux-gnu/usr/lib/ld-linux-riscv64-lp64d.so.1'
    
    # LDFLAGS+=" -L'/apps/riscv/sdv_trace/lib/' -lsdv_trace_rave -lsdv_trace"
    # LDFLAGS+=" -L/apps/riscv/sdv_trace/sdvtrace-development/lib -Wl,-rpath=/apps/riscv/sdv_trace/sdvtrace-development/lib -lsdv_trace"
    LDFLAGS+=" -v"

    if((SDV_TRACING)); then
        LDFLAGS+=" ${SDV_TRACE_C_LINK}"
    fi
fi

export LDFLAGS

targets_d=()
targets_f=()

for p in "${progs[@]}"; do
    base="$(basename "$p")"
    if [[ "$base" =~ _d.exe ]]; then
        targets_d+=("$base")
    elif [[ "$base" =~ _f.exe ]]; then
        targets_f+=("$base")
    fi
done

unset MAKELEVEL

# CPU

if ((${#targets_d[@]} > 0)); then
    export DOUBLE=1
    export CFLAGS="${CFLAGS_D}"
    export CPPFLAGS="${CPPFLAGS_D}"
    export SUFFIX='_d'
    export TARGETS="${targets_d[*]}"
    make -f Makefile_in "$@"
fi

if ((${#targets_f[@]} > 0)); then
    export DOUBLE=0
    export CFLAGS="${CFLAGS_F}"
    export CPPFLAGS="${CPPFLAGS_F}"
    export SUFFIX='_f'
    export TARGETS="${targets_f[*]}"
    make -f Makefile_in "$@"
fi

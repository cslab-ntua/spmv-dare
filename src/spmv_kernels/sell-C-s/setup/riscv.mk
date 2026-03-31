SELLCS_INCLUDES = -I. 
SELLCS_LIBS     = 
LIB_SUFFIX=

# ----------------------------------------------------------------------
# -DINDEX_64               Use 64-bit integer column indices instead of 32-bit
# -DUSE_OMP                Enables OpenMP parallel regions in the code
# -DALIGN_TO=<align_size>  Mallocs will be aligned to 'align_size' bytes. Default: 64
# -DEPI_EXT=v07            Possible values: v07 or empty; If left empty, builds with v0.9 routines

SELLCS_OPTS     =  -DALIGN_TO=1024

# Index data type
ifdef INDEX64
    SELLCS_OPTS += -DINDEX64
    LIB_SUFFIX:=_i64
else
    LIB_SUFFIX:=_i32
endif

# Enable / diable OpenMP
ifdef OMP
    LIB_SUFFIX:=$(LIB_SUFFIX)_omp
    SELLCS_OPTS     += -DUSE_OMP
else
    LIB_SUFFIX:=$(LIB_SUFFIX)_sequential

    ifneq (,$(findstring 07,$(EPI_EXT)))
     SELLCS_OPTS += -DEPI_EXT_07
     LIB_SUFFIX:=$(LIB_SUFFIX)_v07
    endif

    ifneq (,$(findstring 09,$(EPI_EXT)))
        SELLCS_OPTS += -DEPI_EXT_09
        LIB_SUFFIX:=$(LIB_SUFFIX)_v09
    endif
endif


# ----------------------------------------------------------------------

SELLCS_DEFS     = $(SELLCS_OPTS) $(SELLCS_INCLUDES)

# ----------------------------------------------------------------------

# - Compilers / linkers - Optimization flags
# ----------------------------------------------------------------------

CC          ?= clang
AVCC        ?= clang
LLVM        ?= clang
LLVM_FLAGS = $(SELLCS_DEFS) -mepi

ifdef DBG
   CFLAGS     = $(SELLCS_DEFS) -O0 -g -ggdb -DSELLCS_DEBUG $(PARAMS)
   LLVM_FLAGS += -O0 -g -ggdb -DSELLCS_DEBUG
else
   CFLAGS     = $(SELLCS_DEFS) -O3 -DNDEBUG  $(PARAMS)
   LLVM_FLAGS += -O3 -DNDEBUG 
endif

# For some reason.. when setting it to atrevido-vec it gets stuck during execution (only on synth-hca)
CFLAGS	+= -mcpu=avispado 
# CFLAGS  += -mcpu=atrevido-vec

LLVM_FLAGS+= -mcpu=avispado
# LLVM_FLAGS+= -mcpu=atrevido-vec

LLVM_FLAGS+= -fno-lto
LLVM_FLAGS+= -flax-vector-conversions
LLVM_FLAGS+= -Wno-vla-cxx-extension
LLVM_FLAGS+= -fno-slp-vectorize
LLVM_FLAGS+= -mllvm -combiner-store-merging=0
LLVM_FLAGS+= -mllvm -vectorizer-use-vp-strided-load-store
LLVM_FLAGS+= -mllvm -disable-loop-idiom-memcpy
LLVM_FLAGS+= -mllvm -disable-loop-idiom-memset
LLVM_FLAGS+= -mllvm -riscv-uleb128-reloc=0
LLVM_FLAGS+= -Xclang -target-feature -Xclang +does-not-implement-vszext
LLVM_FLAGS+= -Xclang -target-feature -Xclang +does-not-implement-tu
LLVM_FLAGS+= -Rpass=loop-vectorize
LLVM_FLAGS+= -Rpass-analysis=loop-vectorize
LLVM_FLAGS+= -Rpass-missed=loop-vectorize

AVCC_FLAGS = $(LLVM_FLAGS)
ifdef OMP
    CFLAGS += -fopenmp
endif

MASM        = 

LINKER       = $(CC)
LINKFLAGS    = $(CFLAGS) -static

ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo


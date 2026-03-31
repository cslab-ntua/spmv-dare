# arch ?= UNKNOWN
arch ?= riscv
include setup/$(arch).mk
CC=clang

LLVMSRC = sellcs_mv_kernels_epi.c   
LLVMOBJ = $(patsubst %.c,%.o,$(LLVMSRC)) 

BASESRC = radix_sort.c sellcs_format.c sellcs_utils.c sellcs_analyzer.c sellcs_mv.c
BASEOBJ = $(patsubst %.c,%.o,$(BASESRC)) 

# AVSRC = sellcs_mv_autovector.c
# AVOBJ = $(patsubst %.c,%.o,$(AVSRC)) 


LIBNAME = libsellcs-spmv$(LIB_SUFFIX).a 

HEADER_FILES = sellcs-spmv.h

DEST =.

EPI_EXT = 09

.PHONY: clean all install

all:
	rm -f libsellcs-spmv*.a

	make arch=$(arch) EPI_EXT=$(EPI_EXT) CC="$(CC)" LLVM="$(LLVM)" CFLAGS="$(CFLAGS)" $(LIBNAME)
	

libsellcs-spmv_formatonly.a: $(BASEOBJ)
	rm -f $@
	$(ARCHIVER) qv $@ $^
	mkdir -p obj
	mv *.o obj

%.a: $(LLVMOBJ) $(BASEOBJ) #$(AVOBJ)
	rm -f $@
	$(ARCHIVER) qv $@ $^
	mkdir -p obj
	mv *.o obj

$(BASEOBJ): %.o: %.c
	$(CC) -fPIC $(CFLAGS) -I. -o $@ -c $<

$(LLVMOBJ): %.o: %.c
	$(LLVM) -fPIC $(LLVM_FLAGS) -I. -o $@ -c $<

# $(AVOBJ): %.o: %.c
# 	$(AVCC) -fPIC $(AVCC_FLAGS) -I. -o $@ -c $<

clean:
	rm -f obj/*.o *.a


install:
	mkdir -p $(DEST)
	mkdir -p $(DEST)/lib
	mkdir -p $(DEST)/include

	cp libsellcs-spmv*.a $(DEST)/lib
	cp $(HEADER_FILES) $(DEST)/include

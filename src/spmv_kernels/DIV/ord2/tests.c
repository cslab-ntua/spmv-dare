#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <pthread.h>

#include "macros/macrolib.h"
#include "bit_ops.h"

typedef union __attribute__((packed)) { uint8_t  v; }  vec_mask_m32_8_t;

#define vec_mask_m32_8(val)                                ( (vec_mask_m32_8_t)  { .v = val } )

int
main()
{
	double x = 0.923486;
	double y = 0.923493;
	double z = 0.923495;

	uint64_t d = reinterpret_cast(uint64_t, y) - reinterpret_cast(uint64_t, x);
	uint64_t d2 = reinterpret_cast(uint64_t, z) - reinterpret_cast(uint64_t, y);
	uint64_t dord2 = d2 - d;
	uint64_t dord2_xor = d2 ^ d;

	bits_print_bytestream((unsigned char *) &x, sizeof(x));
	bits_print_bytestream((unsigned char *) &y, sizeof(x));
	bits_print_bytestream((unsigned char *) &d, sizeof(x));
	bits_print_bytestream((unsigned char *) &d2, sizeof(x));
	bits_print_bytestream((unsigned char *) &dord2, sizeof(x));
	bits_print_bytestream((unsigned char *) &dord2_xor, sizeof(x));

	return 0;
}


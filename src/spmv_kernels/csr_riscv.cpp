#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "macros/cpp_defines.h"

#include "spmv_kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "time_it_tsc.h"
	#include "parallel_util.h"

	#include <riscv_vector.h>
	
#ifdef __cplusplus
}
#endif


struct thread_data {
	long i_s;
	long i_e;

	long j_s;
	long j_e;

	// ValueType v_s;
	ValueType v_e;
};

static struct thread_data ** tds;


extern int prefetch_distance;


struct CSR : Matrix_Format
{
	INT_T * row_ptr;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	#ifdef EPI_INTRINSICS
	int64_t* localJa; // vvrettos: For riscv vector intrinsics (unused and uninitialized in all other builds)
	#endif

	CSR(INT_T * row_ptr_in, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		int num_threads = omp_get_max_threads();
		double time_balance;

		row_ptr = (typeof(row_ptr)) aligned_alloc(64, (m+1) * sizeof(*row_ptr));
		ja = (typeof(ja)) aligned_alloc(64, nnz * sizeof(*ja));
		a = (typeof(a)) aligned_alloc(64, nnz * sizeof(*a));
		#pragma omp parallel for
		for (long i=0;i<m+1;i++)
			row_ptr[i] = row_ptr_in[i];
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
		{
			a[i]=values[i];
			ja[i]=col_ind[i];
		}

		#ifdef EPI_INTRINSICS
		localJa = (typeof(localJa)) aligned_alloc(64, nnz * sizeof(*localJa));
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
			localJa[i] = (int64_t) ja[i];
		#endif

		tds = (typeof(tds)) aligned_alloc(64, num_threads * sizeof(*tds));

		time_balance = time_it(1,
			_Pragma("omp parallel")
			{
				int tnum = omp_get_thread_num();
				struct thread_data * td;
				int use_processes = atoi(getenv("USE_PROCESSES"));
				td = (typeof(td)) aligned_alloc(64, sizeof(*td));
				tds[tnum] = td;
				if (use_processes)
				{
					loop_partitioner_balance_iterations(num_threads, tnum, 0, m, &td->i_s, &td->i_e);
				}
				else
				{
					#ifdef CUSTOM_X86_VECTOR_PERFECT_NNZ_BALANCE
						long lower_boundary;
						loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &td->j_s, &td->j_e);
						macros_binary_search(row_ptr, 0, m, td->j_s, &lower_boundary, NULL);           // Index boundaries are inclusive.
						td->i_s = lower_boundary;
						_Pragma("omp barrier")
						if (tnum == num_threads - 1)   // If we calculate each thread's boundaries individually some empty rows might be unassigned.
							td->i_e = m;
						else
							td->i_e = td->i_s + 1;
					#else
						loop_partitioner_balance_prefix_sums(num_threads, tnum, row_ptr, m, nnz, &td->i_s, &td->i_e);
						// loop_partitioner_balance(num_threads, tnum, 2, row_ptr, m, nnz, &td->i_s, &td->i_e);
					#endif
				}
			}
		);
		printf("balance time = %g\n", time_balance);
	}

	~CSR()
	{
		free(a);
		free(row_ptr);
		free(ja);
		free(tds);
		#ifdef EPI_INTRINSICS
		free(localJa);
		#endif
	}

	void spmv(ValueType * x, ValueType * y);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr_vector_riscv(CSR * restrict csr, ValueType * restrict x , ValueType * restrict y);
void compute_csr_vector_riscv_bulk(CSR * restrict csr, ValueType * restrict x , ValueType * restrict y);

void
CSR::spmv(ValueType * x, ValueType * y)
{
	#if defined(RISCV_VECTOR_NAIVE)
		compute_csr_vector_riscv(this, x, y);
	#elif defined(RISCV_VECTOR_BULK)
		compute_csr_vector_riscv_bulk(this, x, y);
	#endif
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz, long symmetric, long symmetry_expanded)
{
	if (symmetric && !symmetry_expanded)
		error("symmetric matrices have to be expanded to be supported by this format");
	struct CSR * csr = new CSR(row_ptr, col_ind, values, m, n, nnz);
	// for (long i=0;i<10;i++)
		// printf("%d\n", row_ptr[i]);
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	#if defined(RISCV_VECTOR_NAIVE)
		csr->format_name = (char *) "Custom_CSR_RISCV_VEC";
	#elif defined(RISCV_VECTOR_BULK)
		csr->format_name = (char *) "Custom_CSR_RISCV_VEC_BULK";
	#endif
	return csr;
}


//==========================================================================================================================================
//= CSR RISC-V (vector and bulk)
//==========================================================================================================================================

#ifdef RISCV_VECTOR
/* RISCV_VECTOR */

#ifdef EPI_INTRINSICS
/* EPI Intrinsics */
void
subkernel_csr_vector_riscv(CSR * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e)
{
	ValueType sum;
	long i, j;
	long j_s, j_e; // Column Start, Column End

	/* Main Loop - Thread iterates from rows i_s till i_e */
	for (i = i_s; i < i_e; i++) {
		y[i] = 0;
		j_s = csr->row_ptr[i];
		j_e = csr->row_ptr[i + 1];
		if (j_s == j_e)
			continue;
		sum = 0;

		long nnz = j_e - j_s;

		/* Inner Loop - Iterate over nnz using vector intrinsics */
		long colid;

		for (colid = 0; colid < nnz;) {
			long requestedVectorLength = nnz - colid;
			long givenVectorLength = __builtin_epi_vsetvl(requestedVectorLength, __epi_e64, __epi_m1);

			__epi_1xf64 va = __builtin_epi_vload_1xf64(&csr->a[j_s + colid], givenVectorLength);
			__epi_1xi64 v_idx_col = __builtin_epi_vload_1xi64(&csr->localJa[j_s + colid], givenVectorLength);

			__epi_1xi64 vthree = __builtin_epi_vmv_v_x_1xi64(3, givenVectorLength);

			v_idx_col = __builtin_epi_vsll_1xi64(v_idx_col, vthree, givenVectorLength);

			__epi_1xf64 vx = __builtin_epi_vload_indexed_1xf64(x, v_idx_col, givenVectorLength);

			__epi_1xf64 vprod = __builtin_epi_vfmul_1xf64(va, vx, givenVectorLength);
			__epi_1xf64 partial_res = __builtin_epi_vfmv_v_f_1xf64(0.0, givenVectorLength);
			partial_res = __builtin_epi_vfredsum_1xf64(vprod, partial_res, givenVectorLength);
			sum += __builtin_epi_vfmv_f_s_1xf64(partial_res);

			colid += givenVectorLength;
		}

		y[i] = sum;
	}
}

#else 
/* Risc-V Vector Instrinsics */
void
subkernel_csr_vector_riscv(CSR * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e)
{
	ValueType sum;
	long i;
	long j_s, j_e; // Column Start, Column End

	/* Main Loop - Thread iterates from rows i_s till i_e */
	for (i = i_s; i < i_e; i++) {
		y[i] = 0;
		j_s = csr->row_ptr[i];
		j_e = csr->row_ptr[i + 1];
		if (j_s == j_e)
			continue;
		sum = 0;

		long nnz = j_e - j_s;

		/* Inner Loop - Iterate over nnz using vector intrinsics */
		long colid;

		for (colid = 0; colid < nnz;) {
			long requestedVectorLength = nnz - colid;
			long givenVectorLength = __riscv_vsetvl_e64m1(requestedVectorLength);

			vfloat64m1_t va = __riscv_vle64_v_f64m1(&csr->a[j_s + colid], givenVectorLength);
			vuint32mf2_t v_idx_col = __riscv_vle32_v_u32mf2((uint32_t*) &csr->ja[j_s + colid], givenVectorLength);

			vuint32mf2_t vthree = __riscv_vmv_v_x_u32mf2(3, givenVectorLength);

			v_idx_col = __riscv_vsll_vv_u32mf2(v_idx_col, vthree, givenVectorLength);

			vfloat64m1_t vx = __riscv_vluxei32_v_f64m1(x, v_idx_col, givenVectorLength);

			vfloat64m1_t vprod = __riscv_vfmul_vv_f64m1(va, vx, givenVectorLength);
			vfloat64m1_t partial_res = __riscv_vfmv_v_f_f64m1(0.0, givenVectorLength);
			partial_res = __riscv_vfredusum_vs_f64m1_f64m1(vprod, partial_res, givenVectorLength);
			sum += __riscv_vfmv_f_s_f64m1_f64(partial_res);

			colid += givenVectorLength;
		}

		y[i] = sum;
	}
}

#endif /* EPI_INTRINSICS */

#endif /* RISCV_VECTOR */

#if defined(RISCV_VECTOR_NAIVE)

void
compute_csr_vector_riscv(CSR * restrict csr, ValueType * restrict x, ValueType * restrict y)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		struct thread_data * td = tds[tnum];
		long i_s, i_e;
		i_s = td->i_s;
		i_e = td->i_e;
		subkernel_csr_vector_riscv(csr, x, y, i_s, i_e);
	}
}

#endif


#if defined(RISCV_VECTOR_BULK)
void subkernel_csr_vector_riscv_bulk(CSR * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e)
{

	int maxNaiveComputations = 5000;
	int naiveComputationsCounter = 0;

	ValueType sum;
	// long i;
	long j_s, j_e; // Column Start, Column End

	/* Get the max vector length */
	long maxVectorLength = __riscv_vsetvl_e64m1(0x7FFFFFFF);

	/* Iterate through rows from start to see current step end */
	long currentRow = i_s;
	long rowStart = i_s, rowEnd = i_e;
	long nnz = 0;
	long rowsToCompute = 0;

	/* While givenVectorLength is less than the maxVectorLength (architecture specific), add more rows to the vector */
	while (rowStart < rowEnd) {
		/* Find start and end of current computation part */
		while (nnz < maxVectorLength || currentRow < rowEnd) {
			j_s = csr->row_ptr[rowStart];
			j_e = csr->row_ptr[currentRow + 1];

			if (j_s == j_e) {
				currentRow++;
				continue;
			}

			nnz = j_e - j_s;

			if (nnz >= maxVectorLength) {
				/* Check if we should calculate using the naive method instead */
				rowsToCompute = currentRow - rowStart;

				if (rowsToCompute == 1) {
					/* Non zeroes per row already filled with one row, use naive method instead */					
					if (naiveComputationsCounter >= maxNaiveComputations) {
						subkernel_csr_vector_riscv(csr, x, y, rowStart, rowEnd);
						return;
					}
					else {
						subkernel_csr_vector_riscv(csr, x, y, rowStart, rowStart+1);
						naiveComputationsCounter++;
						rowStart++;
						currentRow++;
						continue;
					}
				}
				/* If we ended up here, that means that we can "efficiently" execute using the other method, meaning that
					multiple rows should be loaded and that we have found our start and end row indices. */
				else {
					break;
				}
			}

			/* We haven't filled the rows yet */
			currentRow++;
		}

		/* I have found that the current j_e - j_s difference is larger than my 
			maxVectorLength, so the endRow of the comp part should be one less than than the currentRow.
			Basically I am doing a mini-kernel between rows i_s=rowStart and i_e=currentRow-1.
		*/

		if (currentRow > rowEnd) {
			currentRow = rowEnd;
		}
		/* Actually Execute Kernel */
		currentRow -= 1;
	
		if (rowStart == currentRow) {
			break;
		}

		j_s = csr->row_ptr[rowStart];
		j_e = csr->row_ptr[currentRow + 1];
		
		nnz = j_e - j_s;

		long givenVectorLength = __riscv_vsetvl_e64m1(nnz);
		if (givenVectorLength != nnz) {
			/* Create a fail-safe for this one as well. Basically make it run the naive vectorization version */
			subkernel_csr_vector_riscv(csr, x, y, rowStart, currentRow - 1);
			rowStart = currentRow;
			continue;
		}

		vfloat64m1_t va = __riscv_vle64_v_f64m1(&csr->a[j_s], givenVectorLength);

		vuint32mf2_t v_idx_col = __riscv_vle32_v_u32mf2((uint32_t*) &csr->ja[j_s], givenVectorLength);

		// luxei expects byte offsets, not indices. so we have to shift left the loaded column indices by 3
		vuint32mf2_t vthree = __riscv_vmv_v_x_u32mf2(3, givenVectorLength);
		v_idx_col = __riscv_vsll_vv_u32mf2(v_idx_col, vthree, givenVectorLength);
		vfloat64m1_t vx = __riscv_vluxei32_v_f64m1(x, v_idx_col, givenVectorLength);

		/* Debug: store vx to array and print contents */
		vfloat64m1_t vprod = __riscv_vfmul_vv_f64m1(va, vx, givenVectorLength);

		/* Loop through rows again. Should keep in an array the number of elements per row to handle the shifts */
		long currentNnz = 0;
		for (long i_sum = rowStart; i_sum < currentRow; i_sum++) {
			currentNnz = csr->row_ptr[i_sum + 1] - csr->row_ptr[i_sum];

			if (currentNnz == 0) {
				y[i_sum] = 0.0;
				continue;
			}

			vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0.0, currentNnz);
			sum = __riscv_vfredusum_vs_f64m1_f64m1(vprod, sum, currentNnz);
			y[i_sum] = __riscv_vfmv_f_s_f64m1_f64(sum);

			/* Shift positions using slidedown */
			vprod = __riscv_vslidedown_vx_f64m1(vprod, currentNnz, givenVectorLength);
		}

		/* Update rowStart */
		rowStart = currentRow;
	}

	return;
}

void
compute_csr_vector_riscv_bulk(CSR * restrict csr, ValueType * restrict x, ValueType * restrict y)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		struct thread_data * td = tds[tnum];
		long i_s, i_e;
		i_s = td->i_s;
		i_e = td->i_e;
		subkernel_csr_vector_riscv_bulk(csr, x, y, i_s, i_e);
	}
}

#endif

//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSR::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSR::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	// int num_threads = omp_get_max_threads();
	// long i, i_s, i_e;
	// for (i=0;i<num_threads;i++)
	// {
		// i_s = tds[i]->i_s;
		// i_e = tds[i]->i_e;
		// printf("%3ld: i=[%8ld, %8ld] (%8ld) , nnz=%8d\n", i, i_s, i_e, i_e - i_s, row_ptr[i_e] - row_ptr[i_s]);
	// }
	return 0;
}


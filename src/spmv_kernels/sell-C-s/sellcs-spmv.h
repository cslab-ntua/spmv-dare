/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/


#ifndef SELLCSVE_H
#define SELLCSVE_H

#include <stdint.h>
#include <stdlib.h>
#include "typedefs.h"

#define UNR 8

extern size_t align_size;

struct SELLCSMatrix_STRUCT
{
    char *name;
    elem_t *values;
    index_t *column_indices;
    index_t nrows;
    index_t ncolumns;
    uint64_t nnz;
    uint64_t C;
    uint64_t sigma;
    index_t nslices;
    index_t *slice_widths;
    index_t *slice_pointers;
    index_t *row_order;
    uint8_t *vop_lengths;
    uint64_t *vop_pointers; // slice widths and vop_pointers can be merged, to express both; just extend vop_pointers size by 1 as row_pointers in csr is nrows + 1;
    uint8_t dfc_enabled;
    index_t *task_ptrs;
    int32_t shift;
};
typedef struct SELLCSMatrix_STRUCT sellcs_matrix_t;

int32_t sellcs_init_params(
    const uint64_t C, // Height of the slice; Most
    const uint64_t sigma_window,
    sellcs_matrix_t *sellcs_mtx);

void free_sellcs(sellcs_matrix_t *sellcs);

void sellcs_execute_mv_d(const sellcs_matrix_t *matrix,
                             const elem_t *restrict x,
                             elem_t *restrict y);

void sellcs_execute_mv_d_autovector(const sellcs_matrix_t *matrix,
                                    const elem_t *restrict x,
                                    elem_t *restrict y);

void sellcs_execute_mv_d_unroll(const sellcs_matrix_t *matrix,
                                const elem_t *restrict x,
                                elem_t *restrict y);

// Analyzer
void sellcs_analyze_matrix(sellcs_matrix_t *matrix,  int32_t use_unroll);

int32_t sellcs_create_matrix_from_CSR_rd(
    const index_t nrow,
    const index_t ncol,
    const index_t *iaptr,
    const index_t *iaind,
    const elem_t *avalues,
    const int32_t indexing,   // Disabled for now, Indexing is always 0-based.
    const int32_t enable_dfc, // enable_dfc = 1 to enable;
    sellcs_matrix_t *sellcs_mtx);


int32_t sellcs_create_matrix_from_ELL_rd(
    const index_t nrow,
    const index_t ncol,
    const index_t *iacol,
    const elem_t *avalues,
    const index_t max_row_size,
    const int column_major,
    const int indexing,             // Set 0 (for 0-based) or 1 (for 1-based) array indexing. 
    const int32_t enable_dfc,       // Set to 1 to create a SELL-c-sigma matrix with DFC optimization data structures.
    sellcs_matrix_t *sellcs_mtx);

int32_t sellcs_create_matrix_from_BCSR_rd(
    const index_t nrows,
    const index_t ncols,
    const index_t *row_bptr,
    const index_t *col_bind,
    const elem_t *avalues,
    const int32_t block_dimension,
    const int32_t indexing,         // Set 0 (for 0-based) or 1 (for 1-based) array indexing. 
    sellcs_matrix_t *sellcs_mtx);

////////////////////////////////////////

struct CSRMatrix_STRUCT
{
    const elem_t *values; // values of matrix entries
    const index_t *column_indices;
    const index_t *row_pointers;
    const index_t nrows;
    const index_t ncolumns;
    uint64_t nnz;
};
typedef struct CSRMatrix_STRUCT csr_matrix_t;



#endif // SELLCSVE_FORMAT_H

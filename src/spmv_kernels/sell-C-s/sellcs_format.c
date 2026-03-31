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

#include "radix_sort.h"
#include "sellcs-spmv.h"
#include "sellcs_utils.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ALIGN_TO
size_t align_size = ALIGN_TO;
#else
size_t align_size = 256;
#endif

int32_t sellcs_init_params(
    const uint64_t C, // Height of the slice; Most
    const uint64_t sigma_window,
    sellcs_matrix_t* sellcs_mtx)
{
    sellcs_mtx->C = C;
    sellcs_mtx->sigma = sigma_window;

    // This is the shift value used in the spmv kernel

    uint64_t tmp = C;
    sellcs_mtx->shift = 0;
    while (tmp >> 1 > 0) {
        sellcs_mtx->shift++;

        tmp = tmp >> 1;
    }
    // fprintf(stderr, "shift: %d \n", sellcs_mtx->shift);
    return 0;
}

void csr_to_sellcs_dfc(const csr_matrix_t* csr_matrix, sellcs_matrix_t* sellcs_mtx)
{

    index_t* rows_size = sellcs_get_rows_size(csr_matrix->row_pointers, csr_matrix->nrows);
    sellcs_mtx->row_order = get_order_by_row_size(rows_size, csr_matrix->nrows, sellcs_mtx->sigma);

    /* Compute slice widths and number of vop (vertical operations) */
    sellcs_mtx->nslices = (csr_matrix->nrows + sellcs_mtx->C - 1) / sellcs_mtx->C;
    sellcs_mtx->slice_widths = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(sellcs_mtx->nslices * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_widths, " SparseMatrixSELLCS.slice_widths");
    sellcs_mtx->vop_pointers = (uint64_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(sellcs_mtx->nslices * sizeof(uint64_t) + 1));
    sellcs_check_malloc(sellcs_mtx->vop_pointers, " SparseMatrixSELLCS.vop_pointers");

    index_t slice_idx = 0;
    uint64_t vop_count = 0;
    for (uint64_t r = 0; r < csr_matrix->nrows; r += sellcs_mtx->C) {
        sellcs_mtx->slice_widths[slice_idx] = rows_size[r];
        sellcs_mtx->vop_pointers[slice_idx] = vop_count;
        vop_count += rows_size[r];
        slice_idx++;
    }
    sellcs_mtx->vop_pointers[slice_idx] = vop_count;

    /* Compute the vop lengths and the size of column_indices and values data structures */
    sellcs_mtx->vop_lengths = (uint8_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(vop_count * sizeof(uint8_t)));
    sellcs_check_malloc(sellcs_mtx->vop_lengths, "SparseMatrixSELLCS.vop_lengths");
    sellcs_set_active_lanes(rows_size, csr_matrix->nrows, sellcs_mtx->C, sellcs_mtx->vop_lengths);

    sellcs_mtx->slice_pointers = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((sellcs_mtx->nslices + 1) * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_pointers, "SparseMatrixSELLCS.slice_pointers");

    uint64_t vop_idx = 0;
    uint64_t size_of_matrix = 0;

    for (uint64_t s = 0; s < sellcs_mtx->nslices; s++) {
        sellcs_mtx->slice_pointers[s] = size_of_matrix;
        for (uint64_t v = 0; v < sellcs_mtx->slice_widths[s]; v++)
            size_of_matrix += sellcs_mtx->vop_lengths[vop_idx++] + 1;
    }
    sellcs_mtx->slice_pointers[sellcs_mtx->nslices] = size_of_matrix;
    uint64_t size_of_values = size_of_matrix * sizeof(elem_t);
    uint64_t size_of_colidx = size_of_matrix * sizeof(index_t);

    /* Allocate SELLCS NNZ data structures in memory */
    sellcs_mtx->column_indices = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_colidx));
    sellcs_check_malloc(sellcs_mtx->column_indices, "SparseMatrixSELLCS.column_indices");

    sellcs_mtx->values = (elem_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_values));
    sellcs_check_malloc(sellcs_mtx->values, "SparseMatrixSELLCS.values");

    /* Copy values and column indices from CSR to SELLCS */
    #pragma omp parallel for schedule(dynamic, 32)
    for (uint64_t s = 0; s < sellcs_mtx->nslices; s++) {
        index_t swidth = sellcs_mtx->slice_widths[s];
        index_t base_row = s * sellcs_mtx->C;
        uint64_t insert_idx = sellcs_mtx->slice_pointers[s];
        uint64_t vop_id = sellcs_mtx->vop_pointers[s];

        for (uint64_t vop = 0; vop < swidth; vop++) {
            for (uint64_t r = 0; r < sellcs_mtx->vop_lengths[vop_id] + 1; r++) // Note:  vop length must be + 1
            {
                uint64_t csr_nnz_idx = csr_matrix->row_pointers[sellcs_mtx->row_order[base_row + r]] + vop;
                sellcs_mtx->values[insert_idx] = csr_matrix->values[csr_nnz_idx];
                sellcs_mtx->column_indices[insert_idx] = csr_matrix->column_indices[csr_nnz_idx];
                insert_idx++;
            }
            vop_id++;
        }
    }

    sellcs_mtx->nnz = csr_matrix->nnz;
    sellcs_mtx->nrows = csr_matrix->nrows;
    sellcs_mtx->ncolumns = csr_matrix->ncolumns;

    free(rows_size);

    // /*  FREE CSR STRUCTURES  */
    // if (freecsr) {
    //     free(csr_matrix->values);
    //     free(csr_matrix->column_indices);
    //     free(csr_matrix->row_pointers);
    // }
}

void csr_to_sellcs(const csr_matrix_t* csr_matrix, sellcs_matrix_t* sellcs_mtx)
{
    index_t* rows_size = sellcs_get_rows_size(csr_matrix->row_pointers, csr_matrix->nrows);
    sellcs_mtx->row_order = get_order_by_row_size(rows_size, csr_matrix->nrows, sellcs_mtx->sigma);

    /* Allocate slices */
    sellcs_mtx->nslices = (csr_matrix->nrows + sellcs_mtx->C - 1) / sellcs_mtx->C;
    sellcs_mtx->slice_widths = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(sellcs_mtx->nslices * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_widths, " SELLCS.slice_widths");
    sellcs_mtx->slice_pointers = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((sellcs_mtx->nslices + 1) * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_pointers, "SELLCS.slice_pointers");
    uint64_t size_of_matrix = 0;

    for (uint64_t s = 0; s < sellcs_mtx->nslices; s++) {
        // Get the Size of the first Row in the Slice.
        index_t max_rsize = rows_size[s * sellcs_mtx->C];

        sellcs_mtx->slice_pointers[s] = size_of_matrix;
        sellcs_mtx->slice_widths[s] = max_rsize;
        size_of_matrix += max_rsize * sellcs_mtx->C;
        // printf("slice = %ld, max_rsize = %ld, size_of_matrix = %.2lfM\n", s, max_rsize, size_of_matrix*1.0/(1000*1000));
    }
    sellcs_mtx->slice_pointers[sellcs_mtx->nslices] = size_of_matrix;
    uint64_t size_of_mvalues = size_of_matrix * sizeof(elem_t);
    uint64_t size_of_mcolidx = size_of_matrix * sizeof(index_t);

    /* Allocate SELLCS Matrix data structure in memory */
    sellcs_mtx->column_indices = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mcolidx));
    sellcs_check_malloc(sellcs_mtx->column_indices, "SELLCS.column_indices");
    memset(sellcs_mtx->column_indices, 0, size_of_mcolidx);

    sellcs_mtx->values = (elem_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mvalues));
    sellcs_check_malloc(sellcs_mtx->values, "SELLCS.values");
    memset(sellcs_mtx->values, 0, size_of_mvalues);

    /*  Adds CSR values and column indices to SELLCS */
    for (uint64_t r = 0; r < csr_matrix->nrows; r++) {
        index_t sidx = r / sellcs_mtx->C;
        // Write at index:
        uint64_t write_idx = sellcs_mtx->slice_pointers[sidx] + (r % sellcs_mtx->C);
        // Read from/to index:
        uint64_t nnz_start = csr_matrix->row_pointers[sellcs_mtx->row_order[r]];
        uint64_t nnz_end = csr_matrix->row_pointers[sellcs_mtx->row_order[r] + 1];

        for (uint64_t i = nnz_start; i < nnz_end; i++) {
            sellcs_mtx->values[write_idx] = csr_matrix->values[i];
            sellcs_mtx->column_indices[write_idx] = csr_matrix->column_indices[i];
            write_idx += sellcs_mtx->C;
        }
    }

    sellcs_mtx->nnz = csr_matrix->nnz;
    sellcs_mtx->nrows = csr_matrix->nrows;
    sellcs_mtx->ncolumns = csr_matrix->ncolumns;

    free(rows_size);

    // /*  FREE CSR STRUCTURES  */
    // if (freecsr) {
    //     free(csr_matrix->values);
    //     free(csr_matrix->column_indices);
    //     free(csr_matrix->row_pointers);
    // }
}

int32_t sellcs_create_matrix_from_CSR_rd(
    const index_t nrows,
    const index_t ncols,
    const index_t* iaptr,
    const index_t* iaind,
    const elem_t* avalues,
    const int32_t indexing, // Disabled for now, Indexing is always 0-based.
    const int32_t enable_dfc,
    sellcs_matrix_t* sellcs_mtx)
{
    const index_t* row_pointers = iaptr;

    // TODO remove CSR struct
    csr_matrix_t csr_matrix = { avalues, iaind, iaptr, nrows, ncols, 0 };

    if (enable_dfc) {
        csr_to_sellcs_dfc(&csr_matrix, sellcs_mtx);
    } else {
        csr_to_sellcs(&csr_matrix, sellcs_mtx);
    }

    return 0;
}

void ell_to_sellcs(const index_t nrows,
    const index_t ncols,
    const index_t* iaind,
    const elem_t* avalues,
    const index_t max_rs,
    const int column_major,
    const int32_t indexing,
    sellcs_matrix_t* sellcs_mtx)
{
    index_t* rs = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));
    sellcs_check_malloc(rs, "rows_size\n");
    memset(rs, 0, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));

    // get row sizes by counting elements != 0.0
    for (int64_t r = 0; r < nrows; r++) {
        rs[r] = max_rs;
        for (int64_t c = max_rs - 1; c >= 0; c--) {
            // Apparently theres matrices with 0.0 sparkled between non-0.0 values.
            // Is it safer to determine the row size by starting at the end of the row.
            if (avalues[c * nrows + r] != 0.0)
                break;

            rs[r]--;
        }
    }
    sellcs_mtx->row_order = get_order_by_row_size(rs, nrows, sellcs_mtx->sigma);

    /* Allocate slices */
    sellcs_mtx->nslices = (nrows + sellcs_mtx->C - 1) / sellcs_mtx->C;
    sellcs_mtx->slice_widths = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(sellcs_mtx->nslices * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_widths, " SELLCS.slice_widths");
    sellcs_mtx->slice_pointers = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((sellcs_mtx->nslices + 1) * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_pointers, "SELLCS.slice_pointers");

    uint64_t size_of_matrix = 0;

    for (uint64_t s = 0; s < sellcs_mtx->nslices; s++) {
        // Get the Size of the first Row in the Slice.
        index_t max_rsize = rs[s * sellcs_mtx->C];

        sellcs_mtx->slice_pointers[s] = size_of_matrix;
        sellcs_mtx->slice_widths[s] = max_rsize;
        size_of_matrix += max_rsize * sellcs_mtx->C;
    }
    sellcs_mtx->slice_pointers[sellcs_mtx->nslices] = size_of_matrix;
    uint64_t size_of_mvalues = size_of_matrix * sizeof(elem_t);
    uint64_t size_of_mcolidx = size_of_matrix * sizeof(index_t);

    /* Allocate SELLCS Matrix data structure in memory */
    sellcs_mtx->column_indices = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mcolidx));
    sellcs_check_malloc(sellcs_mtx->column_indices, "SELLCS.column_indices");
    memset(sellcs_mtx->column_indices, 0, size_of_mcolidx);

    sellcs_mtx->values = (elem_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mvalues));
    sellcs_check_malloc(sellcs_mtx->values, "SELLCS.values");
    memset(sellcs_mtx->values, 0, size_of_mvalues);

    /*  Read ELL values and column indices and write to respective SELLCS*/

    /* Write in COLUMN MAJOR order */
    for (uint64_t r = 0; r < nrows; r++) {
        index_t sidx = r / sellcs_mtx->C;
        // Write at index:
        uint64_t write_idx = sellcs_mtx->slice_pointers[sidx] + (r % sellcs_mtx->C);
        // Read from/to ELL index:
        uint64_t read_row = sellcs_mtx->row_order[r];
        uint64_t read_idx = read_row;

        for (uint64_t i = 0; i < rs[r]; i++) {
            sellcs_mtx->values[write_idx] = avalues[read_idx];
            sellcs_mtx->column_indices[write_idx] = (iaind[read_idx] - indexing);
            write_idx += sellcs_mtx->C;
            read_idx += nrows;
        }
    }

    /* Write in ROW MAJOR order */
    // TODO

    sellcs_mtx->nnz = size_of_matrix;
    sellcs_mtx->nrows = nrows;
    sellcs_mtx->ncolumns = ncols;

    free(rs);
}

int32_t sellcs_create_matrix_from_ELL_rd(
    const index_t nrow,
    const index_t ncol,
    const index_t* iacol,
    const elem_t* avalues,
    const index_t max_row_size,
    const int column_major, // Disabled for now, Storing is always columna major.
    const int indexing, // Disabled for now, Indexing is always 0-based.
    const int32_t enable_dfc,
    sellcs_matrix_t* sellcs_mtx)
{

    if (enable_dfc) {
        // ell_to_sellcs_dfc(&csr_matrix, sellcs_mtx);
    } else {
        ell_to_sellcs(nrow, ncol, iacol, avalues, max_row_size, column_major, indexing, sellcs_mtx);
    }

#ifdef USE_OMP
    sellcs_mtx->task_ptrs = sellcs_get_task_groups(sellcs_mtx, 8, 1);
#endif

    return 0;
}

int32_t sellcs_create_matrix_from_BCSR_rd(
    const index_t nrows,
    const index_t ncols,
    const index_t* brow_ptr,
    const index_t* bcol_ind,
    const elem_t* avalues,
    const int32_t block_dimension,
    const int32_t indexing,
    sellcs_matrix_t* sellcs_mtx)
{

    int32_t block_size = block_dimension * block_dimension;
    //
    index_t* rs = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));
    sellcs_check_malloc(rs, "rows_size\n");
    memset(rs, 0, nrows * sizeof(index_t));

    // get row sizes by counting elements != 0.0
    uint64_t block_idx = 0;

    for (index_t r = 0; r < nrows; r++) {
        for (index_t blk_idx = brow_ptr[r]; blk_idx < brow_ptr[r + 1]; blk_idx++) {
            uint64_t elem_idx = (blk_idx - indexing) * block_size; // Support both 1- and 0-based indexing
            for (index_t blk_col = 0; blk_col < block_dimension; blk_col++) {
                for (index_t blk_row = 0; blk_row < block_dimension; blk_row++) {
                    if (avalues[elem_idx] != (elem_t)0.0)
                        rs[r + blk_row]++;

                    elem_idx++;
                }
            }
        }
    }
    sellcs_mtx->row_order = get_order_by_row_size(rs, nrows, sellcs_mtx->sigma);

    /* Allocate slices */
    sellcs_mtx->nslices = (nrows + sellcs_mtx->C - 1) / sellcs_mtx->C;
    sellcs_mtx->slice_widths = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(sellcs_mtx->nslices * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_widths, " SELLCS.slice_widths");
    sellcs_mtx->slice_pointers = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((sellcs_mtx->nslices + 1) * sizeof(index_t)));
    sellcs_check_malloc(sellcs_mtx->slice_pointers, "SELLCS.slice_pointers");

    uint64_t size_of_matrix = 0;

    for (uint64_t s = 0; s < sellcs_mtx->nslices; s++) {
        // Get the Size of the first Row in the Slice.
        index_t max_rsize = rs[s * sellcs_mtx->C];

        sellcs_mtx->slice_pointers[s] = size_of_matrix;
        sellcs_mtx->slice_widths[s] = max_rsize;
        size_of_matrix += max_rsize * sellcs_mtx->C;
    }
    sellcs_mtx->slice_pointers[sellcs_mtx->nslices] = size_of_matrix;
    uint64_t size_of_mvalues = size_of_matrix * sizeof(elem_t);
    uint64_t size_of_mcolidx = size_of_matrix * sizeof(index_t);

    /* Allocate SELLCS Matrix data structure in memory */
    sellcs_mtx->column_indices = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mcolidx));
    sellcs_check_malloc(sellcs_mtx->column_indices, "SELLCS.column_indices");
    memset(sellcs_mtx->column_indices, 0, size_of_mcolidx);

    sellcs_mtx->values = (elem_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(size_of_mvalues));
    sellcs_check_malloc(sellcs_mtx->values, "SELLCS.values");
    memset(sellcs_mtx->values, 0, size_of_mvalues);

    /*  Reverse order vector*/
    index_t* reverse = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));
    for (size_t r = 0; r < nrows; r++)
        reverse[sellcs_mtx->row_order[r]] = r;

    index_t* written_count = (index_t*)calloc(nrows, sizeof(index_t));
    sellcs_check_malloc(written_count, "SELLCS.values");

    /*  Read ELL values and column indices and write to respective SELLCS*/
    // Suboptimal routine, not expected to be used often.
    for (index_t r = 0; r < nrows; r++) {
        for (index_t blk_idx = brow_ptr[r]; blk_idx < brow_ptr[r + 1]; blk_idx++) {
            uint64_t elem_idx = (blk_idx - indexing) * block_size; // get first element of the block
            index_t colind = bcol_ind[blk_idx] - indexing; // col index of the block

            for (index_t blk_col = 0; blk_col < block_dimension; blk_col++) {
                for (index_t blk_row = 0; blk_row < block_dimension; blk_row++) {
                    index_t write_row = reverse[r + blk_row]; // write_row is the destination SELLCS I have to write
                    index_t sidx = write_row / sellcs_mtx->C; // the slice of write_row
                    index_t row_offset = write_row % sellcs_mtx->C; // the offset within the slice of write_row
                    index_t elem_ofset = written_count[write_row] * sellcs_mtx->C; // how many elements in that row have been written so far
                    uint64_t write_sellcs_idx = sellcs_mtx->slice_pointers[sidx] + row_offset + elem_ofset; // index to write @ sellcs

                    elem_t aval = avalues[elem_idx];
                    if (avalues[elem_idx] != (elem_t)0.0) {
                        sellcs_mtx->values[write_sellcs_idx] = aval;
                        sellcs_mtx->column_indices[write_sellcs_idx] = colind;
                        written_count[write_row]++;
                    }
                    elem_idx++;
                }
                colind++;
            }
        }
    }

    sellcs_mtx->nnz = size_of_matrix;
    sellcs_mtx->nrows = nrows;
    sellcs_mtx->ncolumns = ncols;

    free(rs);
    free(written_count);
    free(reverse);

#ifdef USE_OMP
    sellcs_mtx->task_ptrs = sellcs_get_task_groups(sellcs_mtx, 8, 1);
#endif

    return 0;
}

void free_sellcs(sellcs_matrix_t* sellcs)
{
    free(sellcs->values);
    free(sellcs->column_indices);
    free(sellcs->slice_pointers);
    free(sellcs->slice_widths);
    free(sellcs->row_order);

    if (sellcs->vop_lengths)
        free(sellcs->vop_lengths);

    if (sellcs->vop_pointers)
        free(sellcs->vop_pointers);

#ifdef USE_OMP
    free(sellcs->task_ptrs);
#endif
}

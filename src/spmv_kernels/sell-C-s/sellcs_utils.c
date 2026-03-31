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


#include "sellcs_utils.h"
#include "sellcs-spmv.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>


void sellcs_check_malloc(const void *ptr, const char *err_msg)
{
    if (ptr == NULL)
    {
        fprintf(stderr, "Memory Allocation Error: could not allocate %s. Exiting.\n", err_msg);
        exit(1);
    }
}

index_t *sellcs_get_rows_size(const index_t *__restrict row_ptrs, const index_t nrows)
{
    index_t *rs = (index_t *)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));
    sellcs_check_malloc(rs, "rows_size\n");
    memset(rs, 0, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));

    for (size_t i = 0; i < nrows; i++)
    {
        rs[i] = row_ptrs[i + 1] - row_ptrs[i];
    }

    return rs;
}

/*
 * Returns the closest multiple of [align_size] to [size]
 *
 * @param size Size of memory you want to allocate
 */
size_t sellcs_get_multiple_of_align_size(size_t size)
{
    size_t padded_size = ((align_size - (size % align_size)) % align_size) + size;
    return padded_size;
}

uint64_t sellcs_get_num_verticalops(const index_t *__restrict vrows_size, const index_t nrows, const uint32_t vlen, index_t *__restrict slice_width)
{
    // Pre: rows_size is ordered in descent in blocks of vlen size. Meaning every multiple of 256 in rows_size,
    //      including 0, contains the max row size of that block of rows.
    //      e.g: rows_size[0] contains the highest row size between rows_size[0] to rows_size[0+vlen-1];

    // Post:  Returns the total number of vertical ops required. We need this to allocate the vactive_lanes.

    uint64_t total_vops = 0;
    uint64_t vb_idx = 0;
    for (uint64_t i = 0; i < nrows; i += vlen)
    {
        total_vops += vrows_size[i];
        slice_width[vb_idx++] = vrows_size[i];
    }

    return total_vops;
}

void sellcs_set_active_lanes(const index_t *__restrict vrows_size, const index_t nrows,
                      const uint32_t vlen, uint8_t *__restrict vactive_lanes)
{
    uint64_t vactive_idx = 0;

    for (int64_t i = 0; i < nrows; i += vlen)
    {
        // last_row = min(nrows, i+vlen) - 1;
        uint64_t last_row = ((i + vlen) > nrows) ? (nrows - (uint64_t)1) : (i + vlen - 1);

        // prev_rsize = 0 causes to write first as many vactive_lanes as the minimum row size (which is at vrow_size[end]).
        uint64_t prev_rsize = 0;
        uint64_t lanes_to_disable = 0;

        // current_lanes = min(vlen, end - i); NOTE: You must use current lanes + 1 on the solver.
        uint8_t current_lanes = (uint8_t)(last_row - i);

        // For each row in this slice:
        for (int64_t j = last_row; j >= i; j--)
        {
            // Traverse backwards
            uint64_t current_rsize = vrows_size[j];
            uint64_t diff = current_rsize - prev_rsize;

            if (diff > 0)
            {
                current_lanes -= lanes_to_disable;
                // DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
                for (uint64_t k = 0; k < diff; k++)
                {
                    vactive_lanes[vactive_idx++] = current_lanes;
                }

                lanes_to_disable = 1;
                prev_rsize = current_rsize;
            }
            else
            {
                // DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
                lanes_to_disable++;
            }
        }
    }
}

void sellcs_set_slice_vop_length(const index_t *__restrict rows_size, const index_t slice_height, uint8_t *__restrict vop_lengths)
{
    uint64_t vactive_idx = 0;

    uint64_t last_row = slice_height - 1;
    uint64_t prev_rsize = 0;
    uint64_t lanes_to_disable = 0;

    uint8_t current_lanes = slice_height - 1; // NOTE: You must use current lanes + 1 on the solver.

    for (int64_t j = last_row; j >= 0; j--)
    {
        // Traverse backwards
        uint64_t rsize = rows_size[j];
        uint64_t diff = rsize - prev_rsize;

        if (diff > 0)
        {
            current_lanes -= lanes_to_disable;
            // DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
            // Insert VOPS with the same vop lengths
            for (uint64_t k = 0; k < diff; k++)
            {
                vop_lengths[vactive_idx++] = current_lanes;
            }

            lanes_to_disable = 1;
            prev_rsize = rsize;
        }
        else
        {
            // DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
            lanes_to_disable++;
        }
    }
}


index_t *sellcs_get_task_groups(sellcs_matrix_t *matrix, index_t total_tasks, int32_t unroll)
{
    uint64_t total_nnz = matrix->slice_pointers[matrix->nslices];
    uint64_t nnz_per_task = ((total_nnz + total_tasks - 1) / total_tasks);
    index_t *task_ptrs = (index_t *)malloc((total_tasks + 1) * sizeof(index_t));

    uint64_t task_nnz = 0;
    uint64_t task_idx = 0;
    task_ptrs[task_idx++] = 0;

    for (uint64_t s = 0; s < matrix->nslices; s++)
    {
        if ((task_nnz >= nnz_per_task) && !(s % unroll))
        {
            // fprintf(stderr, "Task #%lu, starts at slice: %lu\n", task_idx, s);
            task_ptrs[task_idx++] = s;
            task_nnz = 0;
        }

        task_nnz += matrix->slice_pointers[s + 1] - matrix->slice_pointers[s];
    }
    task_ptrs[task_idx] = matrix->nslices;

    return task_ptrs;
}


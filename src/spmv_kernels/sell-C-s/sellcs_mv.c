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

#include "sellcs-spmv.h"
#include "sellcs_mv.h"

#ifdef USE_OMP
#include "omp.h"
#endif


#ifndef USE_OMP

/* * * * * * * * * * SEQUENTIAL * * * * * * * * * * */

void sellcs_execute_mv_d(const sellcs_matrix_t *matrix,
                             const elem_t *restrict x,
                             elem_t *restrict y)
{

    sellcs_mv_d(matrix, x, y, 0,  matrix->nslices);

}
    
void sellcs_execute_mv_d_autovector(const sellcs_matrix_t *matrix,
                                    const elem_t *restrict x,
                                    elem_t *restrict y)
{

    sellcs_mv_d_autovector(matrix, x, y, 0, matrix->nrows);

}
    

// void sellcs_execute_mv_d_unroll(const sellcs_matrix_t *matrix,
//                              const elem_t *restrict x,
//                              elem_t *restrict y)
// {
//     sellcs_mv_d_unroll(matrix, x, y, 0,  matrix->nslices);
// }

#else 


void sellcs_execute_mv_d(const sellcs_matrix_t *matrix,
                             const elem_t *restrict x,
                             elem_t *restrict y)
{
  
    #pragma omp parallel
    {
        #pragma omp single
        {
            uint64_t task_idx = 0;
            while (matrix->task_ptrs[task_idx] < matrix->nslices)
            {
                index_t slice_idx = matrix->task_ptrs[task_idx];
                index_t end_slice = matrix->task_ptrs[task_idx + 1];
                #pragma omp task
                sellcs_mv_d(matrix, x, y, slice_idx, end_slice);
                task_idx++;
            }
            #pragma omp taskwait
        }
    }
}


void sellcs_execute_mv_d_unroll(const sellcs_matrix_t *matrix,
                                const elem_t *restrict x,
                                elem_t *restrict y)
{


    #pragma omp parallel
    {
        #pragma omp single
        {
            uint64_t task_idx = 0;
            while (matrix->task_ptrs[task_idx] < matrix->nslices)
            {
                index_t slice_idx = matrix->task_ptrs[task_idx];
                index_t end_slice = matrix->task_ptrs[task_idx + 1];
                #pragma omp task
                sellcs_mv_d(matrix, x, y, slice_idx, end_slice);
                task_idx++;
            }
            #pragma omp taskwait
        }
    }
}
#endif


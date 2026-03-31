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


#include <stddef.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <float.h>

#include "sellcs-spmv.h"
#include "sellcs_mv.h"
#include "sellcs_utils.h"

#include  <stdio.h>

#ifdef USE_OMP
#include "omp.h"

#endif

double sellcs_mytimer(void)
{

    struct timeval tp;
    static long start = 0, startu;
    if (!start)
    {
        gettimeofday(&tp, NULL);
        start = tp.tv_sec;
        startu = tp.tv_usec;
        return 0.0;
    }
    gettimeofday(&tp, NULL);
    return ((double)(tp.tv_sec - start)) + (tp.tv_usec - startu) / 1000000.0;
}

void sellcs_analyze_matrix(sellcs_matrix_t *matrix, int32_t use_unroll)
{
#ifndef USE_OMP
    return;
#else
    double tmp_time, best_time = DBL_MAX;
    int32_t best_partitioning;
    int num_threads = omp_get_num_threads();
    int unroll = 8;

    /* Init aux. x and y vector */
    elem_t *x = (elem_t *)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(matrix->nrows * sizeof(elem_t)));
    sellcs_check_malloc(x, "analyze.x");
    elem_t *y = (elem_t *)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(matrix->nrows * sizeof(elem_t)));
    sellcs_check_malloc(y, "analyze.y");

    for (index_t r = 0; r < matrix->nrows; r++)
        x[r] = r;

    if (use_unroll)
    {
        /* Warmup */
        matrix->task_ptrs = sellcs_get_task_groups(matrix, omp_get_num_threads(), UNR);
        sellcs_execute_mv_d_unroll(matrix, x, y);
        free(matrix->task_ptrs);

        /* Auto-tune best task partitioning */
        for (int32_t i = 1; i < 8; i++)
        {
            index_t total_tasks = i * num_threads;
            matrix->task_ptrs = sellcs_get_task_groups(matrix, total_tasks, UNR);

            /* Run */
            tmp_time = sellcs_mytimer();
            sellcs_execute_mv_d_unroll(matrix, x, y);
            tmp_time = sellcs_mytimer() - tmp_time;

            if (tmp_time < best_time)
                best_partitioning = total_tasks;

            free(matrix->task_ptrs);
        }
        matrix->task_ptrs = sellcs_get_task_groups(matrix, best_partitioning, UNR);
    }
    else
    {
        /* Warmup */
        matrix->task_ptrs = sellcs_get_task_groups(matrix, omp_get_num_threads(), 1);
        sellcs_execute_mv_d_unroll(matrix, x, y);
        free(matrix->task_ptrs);

        /* Auto-tune best task partitioning */
        for (int32_t i = 1; i < 8; i++)
        {
            index_t total_tasks = i * num_threads;
            matrix->task_ptrs = sellcs_get_task_groups(matrix, total_tasks, 1);

            /* Run */
            tmp_time = sellcs_mytimer();
            sellcs_execute_mv_d(matrix, x, y);
            tmp_time = sellcs_mytimer() - tmp_time;

            if (tmp_time < best_time)
                best_partitioning = total_tasks;

            free(matrix->task_ptrs);
        }
        matrix->task_ptrs = sellcs_get_task_groups(matrix, best_partitioning, 1);
    }

    free(x);
    free(y);
#endif
}

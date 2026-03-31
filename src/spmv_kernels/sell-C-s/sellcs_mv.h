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

#ifndef SELLCS_MV_H
#define SELLCS_MV_H

void sellcs_mv_d(const sellcs_matrix_t *matrix,
                 const elem_t *restrict x,
                 elem_t *restrict y,
                 const index_t start_slice,
                 const index_t end_slice);

void sellcs_mv_d_autovector(const sellcs_matrix_t* matrix,
    const elem_t* restrict x,
    elem_t* restrict y,
    const index_t start_row,
    const index_t end_row);


void sellcs_mv_d_unroll(const sellcs_matrix_t *matrix,
                        const elem_t *restrict x,
                        elem_t *restrict y,
                        const index_t start_slice,
                        const index_t end_slice);

#endif // SELLCS_MV_H

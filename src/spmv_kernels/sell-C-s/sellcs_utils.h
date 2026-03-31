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


#ifndef SELLCS_UTILS_H
#define SELLCS_UTILS_H

#include "sellcs-spmv.h"

void sellcs_check_malloc(const void * ptr, const char * err_msg);
size_t sellcs_get_multiple_of_align_size(size_t size);

index_t *sellcs_get_rows_size(const index_t *__restrict row_ptrs, const index_t nrows);
uint64_t sellcs_get_num_verticalops(const index_t *__restrict vrows_size, const index_t nrows, const uint32_t vlen, index_t *__restrict slice_width);

void sellcs_set_active_lanes(const index_t *__restrict vrows_size, const index_t nrows, const uint32_t vlen, uint8_t *__restrict vactive_lanes);
void sellcs_set_slice_vop_length(const index_t *__restrict rows_size, const index_t slice_height, uint8_t *__restrict vop_lengths);

index_t *sellcs_get_task_groups(sellcs_matrix_t *matrix, index_t total_tasks, int32_t unroll);


#endif // UTILS_H

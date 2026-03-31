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
#include "sellcs_utils.h"
#include <inttypes.h>
#include <string.h>

const uint64_t base_p2 = 4; // we will use 2^4 bins
const uint64_t nbins = 1 << base_p2;
const uint64_t base_mod_mask = 0xF; // depends on base_p2

/* RADIX SORT */
// TODO VECTORIZE
uint64_t get_max_value(index_t* order_vector, index_t start, index_t end)
{
    uint64_t max_val = order_vector[start];
    for (size_t i = start + 1; i < end; i++)
        if (order_vector[i] > max_val)
            max_val = order_vector[i];

    return max_val;
}

void count_sort_paired_reversed(
    index_t* order_vector,
    index_t* paired_vector,
    index_t start,
    index_t end,
    uint64_t exp,
    uint64_t* bin_ptrs,
    index_t* tmp_array,
    index_t* tmp_array_paired)
{
    const index_t len = end - start;
    memset(bin_ptrs, 0, nbins * sizeof(uint64_t));

    // Histogram of the CURRENT DIGIT values in ORDER_VECTOR;
    // Shifting by exp removes the previous digits already ordered
    for (uint64_t i = start; i < end; i++) {
        uint64_t bin_id = (order_vector[i] >> exp) & base_mod_mask;
        bin_ptrs[bin_id]++;
    }

    // Indexes to insert values in reverse
    index_t cum_sum = 0;
    for (uint64_t i = 0; i < nbins; i++) {
        cum_sum += bin_ptrs[i];
        bin_ptrs[i] = cum_sum;
    }

    // Insert in order order_vector values to tmp_array ordered.
    for (uint64_t i = start; i < end; i++) {
        const uint64_t bin_id = (order_vector[i] >> exp) & base_mod_mask;
        // Reverse the write index (position)
        const uint64_t write_at_index = len - bin_ptrs[bin_id];
        tmp_array[write_at_index] = order_vector[i];
        tmp_array_paired[write_at_index] = paired_vector[i];
        bin_ptrs[bin_id]--;
    }

    // Copy back ordered values
    memcpy(&order_vector[start], tmp_array, len * sizeof(index_t));
    memcpy(&paired_vector[start], tmp_array_paired, len * sizeof(index_t));
}

void radix_sort_paired_descending(index_t* order_vector, index_t* paired_vector, index_t start, index_t end)
{
    uint64_t* bin_ptrs = (uint64_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nbins * sizeof(uint64_t)));
    // index_t tmp_array[end - start];
    index_t *tmp_array = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((end - start) * sizeof(index_t)));
    index_t *tmp_array_paired = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size((end - start) * sizeof(index_t)));
    // index_t tmp_array_paired[end - start];

    uint64_t max_num_digits = 0;
    uint64_t maxvalue = get_max_value(order_vector, start, end);

    while (maxvalue) {
        max_num_digits++;
        maxvalue = maxvalue >> base_p2;
    }

    uint64_t exp = 0;
    for (uint64_t i = 0; i < max_num_digits; i++) {
        count_sort_paired_reversed(order_vector, paired_vector, start, end, exp, bin_ptrs, tmp_array, tmp_array_paired);
        exp += base_p2;
    }

    free(bin_ptrs);
}

index_t* get_order_by_row_size(index_t* rows_size, const index_t nrows, const size_t sigma_ordering_window)
{
    // Post: rows_size is ordered paired with row_order which contains the offset index (between 0 and sigma_ordering_window-1)
    //       corresponding to the order inside each 'ordering window'.

    index_t* row_order = (index_t*)aligned_alloc(align_size, sellcs_get_multiple_of_align_size(nrows * sizeof(index_t)));
    sellcs_check_malloc(row_order, "get_order_by_row_size.row_order\n");

    // Initialize full row_order between 0 and nrows - 1
    for (uint64_t i = 0; i < nrows; i++) 
        row_order[i] = i;

    #pragma omp parallel for schedule(dynamic, 1)
    for (uint64_t k = 0; k < nrows; k += sigma_ordering_window){
        index_t row_end = (k + sigma_ordering_window > nrows) ? nrows : k + sigma_ordering_window;
        radix_sort_paired_descending(&rows_size[0], &row_order[0], k, row_end);
    }

    return row_order;
}

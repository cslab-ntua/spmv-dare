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


#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <stdint.h>
#include <stdlib.h>

typedef double elem_t;

#ifndef INDEX64
typedef int32_t index_t;
#else
typedef int64_t index_t;
#endif

#endif // SELLCSVE_FORMAT_H

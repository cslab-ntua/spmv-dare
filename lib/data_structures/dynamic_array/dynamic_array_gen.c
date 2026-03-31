#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#include "debug.h"
#include "macros/macrolib.h"
#include "macros/constants.h"
#include "io.h"
#include "lock/lock_util.h"

#include "dynamic_array_gen.h"


//==========================================================================================================================================
//= User Functions Declarations
//==========================================================================================================================================


//==========================================================================================================================================
//= Includes
//==========================================================================================================================================


//==========================================================================================================================================
//= Local Defines
//==========================================================================================================================================


#undef  _TYPE
#define _TYPE  DYNAMIC_ARRAY_GEN_EXPAND_TYPE(_TYPE)
typedef DYNAMIC_ARRAY_GEN_TYPE_1  _TYPE;


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                              Templates                                                                 -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


//==========================================================================================================================================
//= Constructor - Destructor
//==========================================================================================================================================


#undef  dynarray_init
#define dynarray_init  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_init)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
void
dynarray_init(struct dynarray * da, long new_capacity)
{
	da->page_size = GET_PAGE_SIZE();
	if (new_capacity < ((long) sizeof(*da->data)))
		new_capacity = ((long) sizeof(*da->data));
	da->capacity = da->page_size * ((new_capacity + da->page_size - 1) / da->page_size);   // Allocate pages.
	da->max_size = da->capacity / sizeof(*da->data);
	da->size = 0;
	da->data = (typeof(da->data)) safe_mmap_annon(da->capacity);
	da->semaphore = 0;
}


#undef  dynarray_new
#define dynarray_new  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_new)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
struct dynarray *
dynarray_new(long new_capacity)
{
	struct dynarray * da = (typeof(da)) malloc(sizeof(*da));
	dynarray_init(da, new_capacity);
	return da;
}


#undef  dynarray_clean
#define dynarray_clean  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_clean)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
void
dynarray_clean(struct dynarray * da)
{
	safe_munmap(da->data, da->capacity);
}


#undef  dynarray_destroy
#define dynarray_destroy  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_destroy)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
void
dynarray_destroy(struct dynarray ** v_ptr)
{
	dynarray_clean(*v_ptr);
	free(*v_ptr);
	*v_ptr = NULL;
}


//==========================================================================================================================================
//= Resize
//==========================================================================================================================================


#undef  dynarray_resize_base
#define dynarray_resize_base  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_resize_base)
static inline
void
dynarray_resize_base(struct dynarray * da, long new_capacity)
{
	_TYPE * buf;
	if (new_capacity < ((long) sizeof(*da->data)))
		new_capacity = ((long) sizeof(*da->data));
	new_capacity = da->page_size * ((new_capacity + da->page_size - 1) / da->page_size);
	buf = (typeof(buf)) safe_mmap_annon(new_capacity);
	da->data = (typeof(da->data)) safe_mremap_fixed(da->data, da->capacity, new_capacity, buf);
	da->capacity = new_capacity;
}


#undef  dynarray_resize
#define dynarray_resize  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_resize)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
void
dynarray_resize(struct dynarray * da, long new_capacity)
{
	dynarray_resize_base(da, new_capacity);
	da->max_size = da->capacity / sizeof(*da->data);
	if (da->size > da->max_size)
		da->size = da->max_size;
}


//==========================================================================================================================================
//= Clear
//==========================================================================================================================================


// Clear array without deallocating space.
#undef  dynarray_clear
#define dynarray_clear  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_clear)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
void
dynarray_clear(struct dynarray * restrict da)
{
	da->size = 0;
}


//==========================================================================================================================================
//= Set
//==========================================================================================================================================


// Set element at position WITHOUT bounds checking.
#undef  dynarray_set
#define dynarray_set  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_set)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
void
dynarray_set(struct dynarray * restrict da, long pos, _TYPE elem)
{
	da->data[pos] = elem;
}

// Set element at position with bounds checking.
#undef  dynarray_set_safe
#define dynarray_set_safe  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_set_safe)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
void
dynarray_set_safe(struct dynarray * restrict da, long pos, _TYPE elem, _TYPE empty_value)
{
	long new_capacity;
	long new_max_size;
	long i;
	if (__builtin_expect(pos >= da->size, 0))
	{
		if (pos >= da->max_size)
		{
			new_max_size = da->max_size;
			while (pos >= new_max_size)                                    // 'pos' can be much bigger than 'max_size', so we need to iterate.
				new_max_size = 2 * new_max_size;
			new_capacity = new_max_size * sizeof(*da->data);
			dynarray_resize(da, new_capacity);
		}
		for (i=da->size;i<pos;i++)                                             // Initialize unfilled positions to the 'empty_value'.
			da->data[i] = empty_value;
		da->size = pos + 1;
	}
	da->data[pos] = elem;
}


//==========================================================================================================================================
//= Push Back
//==========================================================================================================================================


/* We can only either always use atomic push back, or always use the regular one.
 */


#undef  dynarray_push_back
#define dynarray_push_back  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_push_back)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
long
dynarray_push_back(struct dynarray * restrict da, _TYPE elem)
{
	if (__builtin_expect(da->size >= da->max_size, 0))
		dynarray_resize(da, 2 * da->capacity);
	da->data[da->size++] = elem;
	return da->size - 1;
}


#undef  dynarray_push_back_atomic
#define dynarray_push_back_atomic  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_push_back_atomic)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
long
dynarray_push_back_atomic(struct dynarray * restrict da, _TYPE elem)
{
	long pos = __atomic_fetch_add(&da->size, 1, __ATOMIC_RELAXED);
	while (__builtin_expect(pos >= __atomic_load_n(&da->max_size, __ATOMIC_ACQUIRE), 0))  // Resize.
	{
		if (pos == da->max_size) // It is our responsibility to resize.
		{
			while (pos > __atomic_load_n(&da->semaphore, __ATOMIC_ACQUIRE))        // Wait all previous insertions to complete before copying array.
				lock_cpu_relax();
			dynarray_resize_base(da, 2 * da->capacity);
			__atomic_store_n(&da->max_size, da->capacity / sizeof(*da->data), __ATOMIC_RELEASE);
			break;
		}
		lock_cpu_relax();
	}
	da->data[pos] = elem;
	__atomic_fetch_add(&da->semaphore, 1, __ATOMIC_RELEASE);
	return pos;
}


//==========================================================================================================================================
//= Push Back Array
//==========================================================================================================================================


#undef  dynarray_push_back_array
#define dynarray_push_back_array  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_push_back_array)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
long
dynarray_push_back_array(struct dynarray * restrict da, _TYPE * restrict array, long n)
{
	if (n <= 0)
		return -1;
	if (da->size + n > da->max_size)
	{
		long new_capacity = 2 * da->capacity;
		long bytes = (da->size + n) * sizeof(*da->data);
		while (bytes > new_capacity)
			new_capacity = 2 * new_capacity;
		dynarray_resize(da, new_capacity);
	}
	memcpy(&(da->data[da->size]), array, n * sizeof(*array));
	da->size += n;
	return da->size - n;
}


/* Note: 'data' array is always page-aligned via the mmap allocation. */
#undef  dynarray_push_back_array_aligned
#define dynarray_push_back_array_aligned  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_push_back_array_aligned)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
long
dynarray_push_back_array_aligned(struct dynarray * restrict da, _TYPE * restrict array, long n, long alignment)
{
	if (n <= 0)
		return -1;
	long padding = 0;
	long mod = ((long) da->data + da->size) % alignment;
	if (mod)
		padding = alignment - mod;
	if (da->size + n + padding > da->max_size)
	{
		long new_capacity = 2 * da->capacity;
		long bytes = (da->size + n + padding) * sizeof(*da->data);
		while (bytes > new_capacity)
			new_capacity = 2 * new_capacity;
		dynarray_resize(da, new_capacity);
	}
	memset(&(da->data[da->size]), 0, padding);
	da->size += padding;
	memcpy(&(da->data[da->size]), array, n * sizeof(*array));
	da->size += n;
	return da->size - n;  // Return the start of the inserted data, without the padding.
}


#undef  dynarray_push_back_array_atomic
#define dynarray_push_back_array_atomic  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_push_back_array_atomic)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
inline
long
dynarray_push_back_array_atomic(struct dynarray * restrict da, _TYPE * restrict array, long n)
{
	if (n <= 0)
		return -1;
	long pos = __atomic_fetch_add(&da->size, n, __ATOMIC_RELAXED);
	while (__builtin_expect(pos + n > __atomic_load_n(&da->max_size, __ATOMIC_ACQUIRE), 0))  // Resize.
	{
		if (pos <= da->max_size) // It is our responsibility to resize.
		{
			while (pos > __atomic_load_n(&da->semaphore, __ATOMIC_ACQUIRE))        // Wait all previous insertions to complete before copying array.
				lock_cpu_relax();
			if (pos + n <= __atomic_load_n(&da->max_size, __ATOMIC_ACQUIRE))       // Check again. When max_size changes, multiple threads might pass the 'pos <= max_size'.
				break;
			dynarray_resize_base(da, 2 * da->capacity + n * sizeof(*da->data));
			__atomic_store_n(&da->max_size, da->capacity / sizeof(*da->data), __ATOMIC_RELEASE);
			break;
		}
		lock_cpu_relax();
	}
	memcpy(&(da->data[pos]), array, n * sizeof(*array));
	__atomic_fetch_add(&da->semaphore, n, __ATOMIC_RELEASE);
	return pos;
}


//==========================================================================================================================================
//= Copy Array
//==========================================================================================================================================


#undef  dynarray_copy_to_array
#define dynarray_copy_to_array  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_copy_to_array)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
long
dynarray_copy_to_array(struct dynarray * restrict da,
		_TYPE * array_out)
{
	if (array_out == NULL)
		error("return argument 'array_out' is NULL");
	if (da->size > 0)
		memcpy(array_out, da->data, da->size * sizeof(*array_out));
	return da->size;
}


//==========================================================================================================================================
//= Export Array
//==========================================================================================================================================


#undef  dynarray_export_array
#define dynarray_export_array  DYNAMIC_ARRAY_GEN_EXPAND(dynarray_export_array)
DYNAMIC_ARRAY_GEN_FUNCTION_ATTRIBUTES
long
dynarray_export_array(struct dynarray * restrict da,
		_TYPE ** array_ret)
{
	_TYPE * array = NULL;
	if (array_ret == NULL)
		error("return argument 'array_ret' is NULL");
	if (da->size > 0)
	{
		array = (typeof(array)) malloc(da->size * sizeof(*array));
		memcpy(array, da->data, da->size * sizeof(*array));
	}
	*array_ret = array;
	return da->size;
}


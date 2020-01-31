#pragma once






template <class input_t>
class bufferManagerHost_t
{
public:
	input_t** inputPointer; // array of pointers to the input data (allocated explicitly)
	unsigned int* status;
		#define	statusEmpty 0 // empty (or reuseable)
		#define	statusWrite 1 // host booked for writing
		#define	statusSent 2 // waiting for device read
		#define	statusRead 3 // device booked for reading
	size_t inputPointerIndex; // position of data to add next, incremented by host
	cudaStream_t inputStream; // input data are sent on this stream
	cudaEvent_t dataTransferDone; // to record async memory transfer completion 
public:
	input_t** inputPointer_d; // to pass to the device instance
	unsigned int* status_d; // to pass to the device instance
	size_t capacity;
	cudaError error;

	bufferManagerHost_t(size_t _capacity) : capacity(_capacity)
	{
		// cudaDeviceMapHost = allow use of ZeroCopyMemory 
		cudaSetDeviceFlags(cudaDeviceMapHost);
		// allocate memory for input pointer array & init
		if ((error = cudaHostAlloc(&inputPointer, _capacity * sizeof(input_t*), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&inputPointer_d, inputPointer, 0)) != cudaSuccess) return; // get device pointer
		memset(inputPointer, 0, _capacity * sizeof(input_t*));
		// 	allocate memory for status array & init
		if ((error = cudaHostAlloc(&status, _capacity * sizeof(int), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&status_d, status, 0)) != cudaSuccess) return; // get device pointer
		memset(status, 0, _capacity * sizeof(int));
		// init index
		inputPointerIndex = 0;
		// create stream
		cudaStreamCreate(&inputStream);
		// create event
		if ((error = cudaEventCreate(&dataTransferDone)) != cudaSuccess) return;
	}

	bufferManagerHost_t()
	{
		cudaFreeHost(inputPointer);
		cudaFreeHost(status);
		cudaEventDestroy(dataTransferDone);
	}

	void print()
	{
		printf("bufferManagerHost\n");
		printf("    capcity: %d,  index: %d\n", capacity, inputPointerIndex);
		for (int i = 0; i < capacity; i++)
			printf("    ind %d = ( %p, %d )\n", i, inputPointer[i], status[i]);
		printf("\n");
	}

	int write(input_t** _pointerRef)
	{
		if (status[inputPointerIndex] == statusEmpty)
		{
			status[inputPointerIndex] = statusWrite; // books for writing
			inputPointer[inputPointerIndex] = *_pointerRef; // stores item
			status[inputPointerIndex] = statusSent; // sends to device
			inputPointerIndex = (inputPointerIndex == capacity - 1) ? 0 : inputPointerIndex + 1; // increment index
			return 0; // succesful
		}
		return -1; // can't write
	}

	void stop(int _times) // sends stop signals, if required, waits for empty positions
	{
		for (int i = 0; i < _times; i++)
		{
			while (status[inputPointerIndex] != statusEmpty); // wait until pos is available
			status[inputPointerIndex] = statusWrite; // books for writing
			inputPointer[inputPointerIndex] = (input_t*)(-1); // stores stop signal
			status[inputPointerIndex] = statusSent; // sends to device
			inputPointerIndex = (inputPointerIndex == capacity - 1) ? 0 : inputPointerIndex + 1; // increment index
		}
	}
};




template <class input_t>
class bufferManagerDevice_t
{
public:
	volatile input_t** inputPointer; // array of pointers to the input data (allocated explicitly)
	volatile unsigned int* status; // array of mutex locks
	size_t capacity;
	volatile size_t inputPointerIndex; // position of next data to read, incremented by the device workgroup, who took the input
public:
	__device__ bufferManagerDevice_t(volatile input_t*** _inputPointerRef, volatile unsigned int** _statusRef, size_t _capacity)
	{
		// store device pointers read from the host instance
		inputPointer = *_inputPointerRef;
		status = *_statusRef;
		// set index
		capacity = _capacity;
		inputPointerIndex = 0;
	}

	__device__ void print()
	{
		printf("bufferManagerDevice\n");
		printf("    capcity: %d,  index: %d\n", capacity, inputPointerIndex);
		for (int i = 0; i < capacity; i++)
			printf("    ind %d = ( %p, %d )\n", i, inputPointer[i], status[i]);
		printf("\n");
	}

	__forceinline__ __device__ input_t* read()
	{
		if (status[inputPointerIndex] != statusSent) return NULL; // unsuccessful read (no writing -> allows mem transfer)
		volatile unsigned int* thisStatusRef = &(status[inputPointerIndex]); // read status address (must be volatile, otherwise write to cache only)
		int oldStatus = atomicCAS(const_cast<unsigned int*>(thisStatusRef), statusSent, statusRead); // try to book for reading
		if (oldStatus == statusSent) // booking was succesful -> no other thread can change this item or its status
		{
			input_t* result = const_cast<input_t*>(inputPointer[inputPointerIndex]); // read the item
			inputPointerIndex = (inputPointerIndex == capacity - 1) ? 0 : inputPointerIndex + 1; // increment index
			*thisStatusRef = statusEmpty; // indicates, that this input has used
			return result; // returns pointer
		}
		return NULL; // unsuccessful read - maybe after the first check another thread was faster in booking
	}
};






























/*

syntax: cudaTaskList_t<long long int, long int> anyTaskList(1000);

task list for cuda kernels to store tasks to execute
- where tasks are identified by an id (can be used to index inputs / output arrays)

it can handle parallel access as the following operations are based on atomic operations
- locking a list item to add a task, then filling data
- locking a list item to start processing (execution of task), then indicate it as done

*/

class lock
{
	int* mutex;

	__host__ lock() 
	{
		int state = 0; // temp var
		cudaMalloc((void**) &mutex, sizeof(int)); // alloc on device
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice); // copy 0 value to device
	}

	__host__ ~lock()
	{
		cudaFree(mutex);
	}

	__device__ void lock() 
	{
		while(atomicCAS(mutex, 0, 1) != 0);
	}

	__device__ void unlock() 
	{
		atomicExch(mutex, 0);
	}
};







/*


template<class inputItem_t, class outputItem_t, class index_t>
class dataDispatcher_t
{
private:
	mappedMemoryArray<inputItem_t*, index_t>* inputItemPointerArray; // device pointers to inputs provided for the kernel 
	mappedMemoryArray<outputItem_t*, index_t>* outputItemPointerArray; // device pointers to outputs, given by the kernel
	mappedMemoryArray<index_t, int>* indexArray;
	// index[ ] = {capacity; length; lower; upper}

	int mutex; // 0 = not locked, 1 = locked (on device side atomically switched to avoid racing conditions)
	cudaError error;
public:
	__host__ dataDispatcher_t(index_t _capacity)
	{
		if ((error = cudaMallocManaged(&data, _capacity * sizeof(item_t))) != cudaSuccess) return; // managed memory allocation

		capacity = _capacity;
		length = 0;
		lower = 0;
		upper = 0;
		mutex = 0;
	}

	__host__ ~dataDispatcher_t()
	{
		cudaFree((void*)data);
	}

	__host__ __forceinline__ __device__ void show()
	{
		printf("[ ");
		for (index_t m = 0, i = lower; m < length; ++m, i = ((i + 1) == capacity ? 0 : i + 1))
			printf("%d, ", data[i]);
		printf("]");
	}


	__host__ __forceinline__ __device__ item_t& get(index_t i)
	{
		assert(i < length);
		i += lower;
		i = i > (capacity - 1) ? i - capacity : i;
		return data[i];
	}

	__host__ __forceinline__ __device__ item_t& operator[] (index_t i)
	{
		return get(i);
	}

	__host__ void lock_h()
	{
		while (mutex); // waits until locked
		mutex = 1;
	}

	__host__ void unlock_h()
	{
		mutex = 0;
	}


	__device__ void lock_d()
	{
		while (atomicCAS(&mutex, 1, 1)); // busy wait until locked
		atomicExch(&mutex, 1); // locks
	}

	__device__ void unlock_d()
	{
		atomicExch(&mutex, 0);
	}

	__host__ void push_front_h(item_t value)
	{
		lock_h();
		assert(length + 1 <= capacity);
		lower = (lower == 0) ? capacity - 1 : lower - 1;
		data[lower] = value;
		length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_front_d(item_t value)
	{
		lock_d();
		assert(length + 1 <= capacity);
		lower = (lower == 0) ? capacity - 1 : lower - 1;
		data[lower] = value;
		length += 1;
		unlock_d();
	}


	__host__ void push_back_h(item_t value)
	{
		lock_h();
		assert(length + 1 <= capacity);
		data[upper] = value;
		upper = (upper == capacity - 1) ? 0 : upper + 1;
		length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_back_d(item_t value)
	{
		lock_d();
		assert(length + 1 <= capacity);
		data[upper] = value;
		upper = (upper == capacity - 1) ? 0 : upper + 1;
		length += 1;
		unlock_d();
	}

	__host__ item_t pop_front_h()
	{
		lock_h();
		assert(length > 0);
		item_t result = data[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_h();
		return result;
	}

	__forceinline__ __device__ item_t pop_front_d()
	{
		lock_d();
		assert(length > 0);
		item_t result = data[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_d();
		return result;
	}

	__host__ item_t pop_back_h()
	{
		lock_h();
		assert(length > 0);
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_h();
		return data[upper];
	}

	__forceinline__ __device__ item_t pop_back_d()
	{
		lock_d();
		assert(length > 0);
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_d();
		return data[upper];
	}

};

*/



// **********************  


template<class item_t, class index_t>
class mappedMemoryList_t
{
private:
	item_t* data_h; // array of stored items
	item_t* data_d;
	index_t* index_h; // index[ ] = {capacity; length; lower; upper}
	index_t* index_d;
	int* mutex_h; // 0 = not locked, 1 = locked (on device side atomically switched to avoid racing conditions on mutex)
	int* mutex_d;

	cudaError error;
public:

	#define capacity_h (index_h[0])
	#define length_h   (index_h[1])
	#define lower_h    (index_h[2])
	#define upper_h    (index_h[3])
	#define capacity_d (index_d[0])
	#define length_d   (index_d[1])
	#define lower_d    (index_d[2])
	#define upper_d    (index_d[3])

	__host__ mappedMemoryList_t(index_t _capacity)
	{
		// data
		if ((error = cudaHostAlloc(&data_h, _capacity * sizeof(item_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&data_d, data_h, 0)) != cudaSuccess) return; // get device pointer
		// index
		if ((error = cudaHostAlloc(&index_h, 4 * sizeof(index_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&index_d, index_h, 0)) != cudaSuccess) return; // get device pointer
		// mutex
		if ((error = cudaHostAlloc(&mutex_h, sizeof(int), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&mutex_d, mutex_h, 0)) != cudaSuccess) return; // get device pointer
		// initialize
		*mutex_h = 0;
		capacity_h = _capacity; 
		length_h = 0; 
		lower_h = 0;
		upper_h = 0; 
	}

	__host__ ~mappedMemoryList_t()
	{
		cudaFreeHost(data_h);
		cudaFreeHost(index_h);
		cudaFreeHost(mutex_h);
	}

	__host__ void show_h()
	{
		printf("[ ");
		for (index_t m = 0, i = lower_h; m < length_h; ++m, i = ((i + 1) == capacity_h ? 0 : i + 1))
			printf("%d, ", data_h[i]);
		printf("]");
	}

	__forceinline__ __device__ void show_d()
	{
		printf("[ ");
		for (index_t m = 0, i = lower_d; m < length_d; ++m, i = ((i + 1) == capacity_d ? 0 : i + 1))
			printf("%d, ", data_d[i]);
		printf("]");
	}

	__host__ item_t& get_h(index_t i)
	{
		assert(i < length_h);
		i += lower_h;
		i = i > (capacity_h - 1) ? i - capacity_h : i;
		return data_h[i];
	}

	__host__ __forceinline__ __device__ item_t& get_d(index_t i)
	{
		assert(i < length_h);
		i += lower_h;
		i = i > (capacity_h - 1) ? i - capacity_h : i;
		return data_d[i];
	}

	__host__ void lock_h()
	{
		while (*mutex_h); // waits until mutex == 0 (unlocked) 
		*mutex_h = 1;
	}

	__host__ void unlock_h()
	{
		*mutex_h = 0;
	}

	__forceinline__ __device__ void lock_d()
	{
		while (atomicExch(mutex_d, 1)); // reads  mutex to "old", changes to 1 and repeats this until "old" == 0 (unlocked)
	}

	__forceinline__ __device__ void unlock_d()
	{
		atomicExch(mutex_d, 0);
	}

	__host__ void push_front_h(item_t value)
	{
		lock_h();
		assert(length_h + 1 <= capacity_h);
		lower_h = (lower_h == 0) ? capacity_h - 1 : lower_h - 1;
		data_h[lower_h] = value;
		length_h += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_front_d(item_t value)
	{
		lock_d();
		assert(length_d + 1 <= capacity_d);
		lower_d = (lower_d == 0) ? capacity_d - 1 : lower_d - 1;
		data_d[lower_d] = value;
		length_d += 1;
		unlock_d();
	}

	__host__ void push_back_h(item_t value)
	{
		lock_h();
		assert(length_h + 1 <= capacity_h);
		data_h[upper_h] = value;
		upper_h = (upper_h == capacity_h - 1) ? 0 : upper_h + 1;
		length_h += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_back_d(item_t value)
	{
		lock_d();
		assert(length_d + 1 <= capacity_d);
		data_d[upper_d] = value;
		upper_d = (upper_d == capacity_d - 1) ? 0 : upper_d + 1;
		length_d += 1;
		unlock_d();
	}

	__host__ item_t pop_front_h()
	{
		lock_h();
		assert(length_h > 0);
		item_t result = data_d[lower_h];
		lower_h = (lower_h == capacity_h - 1) ? 0 : lower_h + 1;
		length_h -= 1;
		unlock_h();
		return result;
	}

	__forceinline__ __device__ item_t pop_front_d()
	{
		lock_d();
		assert(length_d > 0);
		item_t result = data_d[lower_d];
		lower_d = (lower_d == capacity_d - 1) ? 0 : lower_d + 1;
		length_d -= 1;
		unlock_d();
		return result;
	}

	__host__ item_t pop_back_h()
	{
		lock_h();
		assert(length_h > 0);
		upper_h = (upper_h == 0) ? capacity_h - 1 : upper_h - 1;
		length_h -= 1;
		unlock_h();
		return data_h[upper_h];
	}

	__forceinline__ __device__ item_t pop_back_d()
	{
		lock_d();
		assert(length_d > 0);
		upper_d = (upper_d == 0) ? capacity_d - 1 : upper_d - 1;
		length_d -= 1;
		unlock_d();
		return data_d[upper_d];
	}

	#undef capacity_h 
	#undef length_h  
	#undef lower_h 
	#undef upper_h  
	#undef capacity_d 
	#undef length_d  
	#undef lower_d 
	#undef upper_d  
};
   


// *********************************



template<class item_t, class index_t>
class mappedMemoryList_t
{
private:
	item_t* data_h; // array of stored items
	item_t* data_d;
	index_t capacity, length, lower, upper;
	int mutex; // 0 = not locked, 1 = locked (on device side atomically switched to avoid racing conditions on mutex)
	cudaError error;
public:
	__host__ void construct(index_t _capacity)
	{
		if ((error = cudaHostAlloc(&data_h, _capacity * sizeof(item_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&data_d, data_h, 0)) != cudaSuccess) return; // get device pointer
		mutex = 0;
		capacity = _capacity;
		length = 0;
		lower = 0;
		upper = 0;
	}

	__host__ void destruct()
	{
		cudaFreeHost(data_h);
	}

	__host__ void show_h()
	{
		printf("[ ");
		for (index_t m = 0, i = lower; m < length; ++m, i = ((i + 1) == capacity ? 0 : i + 1))
			printf("%d, ", data_h[i]);
		printf("]");
	}

	__forceinline__ __device__ void show_d()
	{
		printf("[ ");
		for (index_t m = 0, i = lower; m < length; ++m, i = ((i + 1) == capacity ? 0 : i + 1))
			printf("%d, ", data_d[i]);
		printf("]");
	}

	__host__ item_t& get_h(index_t i)
	{
		assert(i < length);
		i += lower;
		i = i > (capacity - 1) ? i - capacity : i;
		return data_h[i];
	}

	__host__ __forceinline__ __device__ item_t& get_d(index_t i)
	{
		assert(i < length);
		i += lower;
		i = i > (capacity - 1) ? i - capacity : i;
		return data_d[i];
	}

	__host__ void lock_h()
	{
		while (mutex); // waits until mutex == 0 (unlocked) 
		mutex = 1;
	}

	__host__ void unlock_h()
	{
		mutex = 0;
	}

	__forceinline__ __device__ void lock_d()
	{
		while (atomicExch(&mutex, 1)); // reads  mutex to "old", changes to 1 and repeats this until "old" == 0 (unlocked)
	}

	__forceinline__ __device__ void unlock_d()
	{
		atomicExch(&mutex, 0);
	}

	__host__ void push_front_h(item_t value)
	{
		lock_h();
		assert(length + 1 <= capacity);
		lower = (lower == 0) ? capacity - 1 : lower - 1;
		data_h[lower] = value;
		length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_front_d(item_t value)
	{
		lock_d();
		assert(length + 1 <= capacity);
		lower = (lower == 0) ? capacity - 1 : lower - 1;
		data_d[lower] = value;
		length += 1;
		unlock_d();
	}

	__host__ void push_back_h(item_t value)
	{
		lock_h();
		assert(length + 1 <= capacity);
		data_h[upper] = value;
		upper = (upper == capacity - 1) ? 0 : upper + 1;
		length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_back_d(item_t value)
	{
		lock_d();
		assert(length + 1 <= capacity);
		data_d[upper] = value;
		upper = (upper == capacity - 1) ? 0 : upper + 1;
		length += 1;
		unlock_d();
	}

	__host__ item_t pop_front_h()
	{
		lock_h();
		assert(length > 0);
		item_t result = data_h[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_h();
		return result;
	}

	__forceinline__ __device__ item_t pop_front_d()
	{
		lock_d();
		assert(length > 0);
		item_t result = data_d[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_d();
		return result;
	}

	__host__ item_t pop_back_h()
	{
		lock_h();
		assert(length > 0);
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_h();
		return data_h[upper];
	}

	__forceinline__ __device__ item_t pop_back_d()
	{
		lock_d();
		assert(length > 0);
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_d();
		return data_d[upper];
	}

};









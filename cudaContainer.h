
#pragma once




/*

global memory array class 
-------------------------
for device ONLY use!
- lifetime can span kernel multiple lauches
- allocates on device side ONLY
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class globalMemoryArray_t
{
public:
	item_t* data_d;
	index_t capacity;
	cudaError error;
public:
	globalMemoryArray_t(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaMalloc(&data_d, capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the device
	}

	~globalMemoryArray_t(void)
	{
		cudaFree(data_d);
	}
};


/*

pinned (page-locked) memory array class 
---------------------------------------
for device to host, host to device async transfers 
allocates on host and device side
- allocated area is cached on GPU, multiple device side READ / WRITE is possible.
- fast, but transfer is defined explicitly 
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class pinnedMemoryArray_t
{
public:
	item_t* data_h;
	item_t* data_d;
	index_t capacity;
	cudaError error;
public:
	pinnedMemoryArray_t(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaMallocHost(&data_h, capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the host 
		if ((error = cudaMalloc(&data_d, capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the device
	}

	~pinnedMemoryArray_t(void)
	{
		cudaFreeHost(data_h);
	}

	void copyToHost(cudaStream_t _stream, cudaEvent_t _event)
	{
		error = cudaMemcpyAsync(data_h, data_d, capacity * sizeof(item_t), cudaMemcpyDeviceToHost, _stream);
		cudaEventRecord(_event, _stream);
	}

	void copyToDevice(cudaStream_t _stream, cudaEvent_t _event)
	{
		error = cudaMemcpyAsync(data_d, data_h, capacity * sizeof(item_t), cudaMemcpyHostToDevice, _stream);
		cudaEventRecord(_event, _stream);
	}
};



/*

mapped (zero copy) memory array class
-------------------------------------
async, always accessed with PCI-Express’s low bandwidth and high latency -> slow
useful when the device has no sufficient memory (only fragments are transferred)
- device side READ or WRITE is possible only ONCE! - sure?  zero copy is used for communication between host & device
- use for small amount of data, at high occupancy to hide PCIe latency
- coalescing is critically important
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class mappedMemoryArray_t
{
public:
	item_t* data_h;
	item_t* data_d;
	index_t capacity;
	cudaError error;
public:
	mappedMemoryArray_t(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaHostAlloc(&data_h, capacity * sizeof(item_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&data_d, data_h, 0)) != cudaSuccess) return; // get device pointer
	}

	~mappedMemoryArray_t(void)
	{
		cudaFreeHost(data_h);
	}

};



/*

managed (unified) memory array class
-------------------------------------
host & device side transparently accessible array
- avoid device side concurent access of the same elements!
- avoid host-device concurent access of the same elements!

my experience is, that data migration (if necessary) happens only BEFORE and AFTER kernel run
this memory model is NOT useful for host-device communication 

*/

template<class item_t, class index_t> 
class managedMemoryArray_t
{
public:
	item_t* data;
	index_t capacity;
	cudaError error;
public:
	managedMemoryArray_t(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaMallocManaged(&data, capacity * sizeof(item_t))) != cudaSuccess) return; // managed memory allocation
	}

	~managedMemoryArray_t(void)
	{
		cudaFree((void*)data);
	}

};







/*

mapped (zero copy) memory list class
------------------------------------
create an mapped mem instance on host & device side
then call "construct" to allocate internal array and initialise indexes 
before freeing call destruct to free internal allocations

provides a two end container with mapped memory item array allocation
the item references work on both host and device side

to avoid device-device and device-host racing conditions 
a mutex locking mechanism has implemented

frequently alternating host / device  access destroys the performance

*/





template<class item_t, class index_t>
class mappedMemoryList_t
{
private:
	item_t* data_h; // array of stored items
	item_t* data_d;
	index_t capacity;
	index_t length;
	index_t lower;
	index_t upper;
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

	__host__ void show()
	{
		printf("[ ");
		for (index_t m = 0, i = lower; m < length; ++m, i = ((i + 1) == capacity ? 0 : i + 1))
			printf("%d, ", data_h[i]);
		printf("]");
	}

	__host__ index_t length_h()
	{
		return length;
	}

	__forceinline__ __device__ index_t length_d()
	{
		return length;
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
		if (length <= 0) return 999;
		lock_h();
		item_t result = data_h[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_h();
		return result;
	}

	__forceinline__ __device__ item_t pop_front_d()
	{
		if (length <= 0) return 999;
		lock_d();
		item_t result = data_d[lower];
		lower = (lower == capacity - 1) ? 0 : lower + 1;
		length -= 1;
		unlock_d();
		return result;
	}

	__host__ item_t pop_back_h()
	{
		if (length <= 0) return 999;
		lock_h();
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_h();
		return data_h[upper];
	}

	__forceinline__ __device__ item_t pop_back_d()
	{
		if (length <= 0) return 999;
		lock_d();
		upper = (upper == 0) ? capacity - 1 : upper - 1;
		length -= 1;
		unlock_d();
		return data_d[upper];
	}

};



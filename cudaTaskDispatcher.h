#pragma once

#include <stdio.h>
#include "cuda_runtime.h"


// statuses - do not change the numbers!
#define	statusStop	 0 // block must stop
#define	statusHost   1 
#define	statusDevice 2 
#define	statusCopy   4  
#define	statusFree   8 

/*
input HOST status
	[status]		[activity]
	device*			host get()
	host			host can write
	host			host tries send() -> only if device status == free -> copy H2D -> set device status = host
	device ->

input DEVICE status
	free*
	host			device tries get() -> only if host status == device/stop
	device			device can read. 
	device			device send()
	free ->

output HOST status
	[status]		[activity]
	device*			host tries get() -> only if device status == free -> copy D2H -> set device status = host
	host			host can read
	host			host send()
	device ->

output DEVICE status
	[status]		[activity]
	host*			device tries get() -> only if host status == device
	device			device can write
	device			device send()
	free


*/

// ************************************* input, host

template <class item_t, class index_t>
class inputDispatcherHost_t
{
public:
	index_t arraySize; // amount of (item_t) data in one input array
	item_t** hostArray; // array of host pointers to the input data arrays
	item_t** deviceArray; // array of device pointers to the input data arrays
	item_t** deviceArray_d;

	int bufferSize; // how many arrays to manage
	int* hostStatus; // the status of the host side data arrays
	int* hostStatus_d;
	int* deviceStatus; // the status of the device side data arrays
	int* deviceStatus_d;

	unsigned int index; // index of actual  array
	cudaStream_t stream; // data are sent on this stream
	cudaEvent_t transferDone; // to record async memory transfer completion 

public:
	inputDispatcherHost_t(int _bufferSize, index_t _arraySize) : bufferSize(_bufferSize), arraySize(_arraySize)
	{
		cudaStreamCreate(&stream);
		cudaEventCreate(&transferDone);
		// init index
		index = 0;
		// allocate memory for array of pointers
		cudaMallocHost(&hostArray, bufferSize * sizeof(item_t*)); // allocates on the host 
		cudaMallocHost(&deviceArray, bufferSize * sizeof(item_t*)); // allocates on the host 
		cudaMalloc(&deviceArray_d, bufferSize * sizeof(item_t*)); // allocates on the device
		// allocate pinned memory for the arrays
		for (int i = 0; i < bufferSize; i++)
		{
			cudaMallocHost(&hostArray[i], arraySize * sizeof(item_t)); // allocates on the host 
			cudaMalloc(&deviceArray[i], arraySize * sizeof(item_t)); // allocates on the device
		}
		// copy device side array pointers to the device (only once must be done as pointers do not change) 
		cudaMemcpyAsync(deviceArray_d, deviceArray, bufferSize * sizeof(item_t*), cudaMemcpyHostToDevice, stream);
		cudaEventRecord(transferDone, stream);
		// 	allocate mapped memory for status array
		cudaHostAlloc(&hostStatus, _bufferSize * sizeof(int), cudaHostAllocMapped); // allocate on host side and map on device side
		cudaHostGetDevicePointer(&hostStatus_d, hostStatus, 0); // get device pointer
		cudaHostAlloc(&deviceStatus, _bufferSize * sizeof(int), cudaHostAllocMapped); // allocate on host side and map on device side
		cudaHostGetDevicePointer(&deviceStatus_d, deviceStatus, 0); // get device pointer
		// init statuses
		for (int i = 0; i < bufferSize; i++)
		{
			hostStatus[i] = statusDevice; 
			deviceStatus[i] = statusFree; 
		}	
		// wait till pointer transfer is finished
		cudaEventSynchronize(transferDone);
	}

	~inputDispatcherHost_t()
	{
		for (int i = 0; i < bufferSize; i++)
			cudaFreeHost(hostArray[i]);
		cudaFreeHost(hostArray);
		cudaFreeHost(deviceArray);
		cudaFreeHost(hostStatus);
		cudaFreeHost(deviceStatus);
		cudaEventDestroy(transferDone);
		cudaStreamDestroy(stream);
	}

	void print()
	{
		printf("inputDispatcherHost\n");
		printf("    buffer size: %d  at %p\n", bufferSize, hostArray);
		printf("    array size : %d  , at ...\n", arraySize);
		for (int i = 0; i < bufferSize; i++) 
		{
			printf("    [%d] at h:%p  d:%p,    status h:%d  d:%d  ", i, hostArray[i], deviceArray[i], hostStatus[i], deviceStatus[i]);
			for (int j = 0; j < 300; j++)
				printf("<%d>", hostArray[i][j]);
			printf("\n\n");
		}
	}

	item_t* operator[](int _arrayId)
	{
		return hostArray[_arrayId];
	}

	int send() // send actual if possible, and step index
	{
		if (deviceStatus[index] == statusFree)
		{ 
			cudaMemcpyAsync(deviceArray[index], hostArray[index], arraySize * sizeof(item_t), cudaMemcpyHostToDevice, stream);
			cudaEventRecord(transferDone, stream);
			cudaEventSynchronize(transferDone); // wait till copy is finished
			hostStatus[index] = statusDevice; // host can reuse
			deviceStatus[index] = statusHost; // device can get
			index = (index == bufferSize - 1) ? 0 : index + 1; // step to next array
			return 0; // succesful
		}
		return -1; // unsuccesful
	}

	int get(int* _arrayIdRef) // get the index of actual
	{
		if (hostStatus[index] == statusDevice)
		{ 
			*_arrayIdRef = index; // passes the value
			hostStatus[index] = statusHost;
			return 0; // succesful
		}
		return -1; // unsuccesful: the status was not OK
	}

	void sendStop(int _times) // sends stop signals
	{
		for (int i = 0; i < _times; i++)
		{
			while (deviceStatus[index] != statusFree) ; // wait until becomes available
			hostStatus[index] = statusStop; // safety -> host can't use
			deviceStatus[index] = statusHost; // send stop signal
			index = (index == bufferSize - 1) ? 0 : index + 1; // increment index	
		}
	}
};


// ************************************* input, device

template <class item_t, class index_t>
class inputDispatcherDevice_t
{
public:
	index_t arraySize; // amount of (item_t) data in one input array
	item_t** deviceArray;

	int bufferSize; // how many arrays to manage
	volatile int* hostStatus;
	volatile int* deviceStatus;

	volatile unsigned int index; // index of actual  array

public:
	__device__ void setup(
		size_t _arraySize, 
		void*** _deviceArrayRef,
		int _bufferSize,
		volatile int** _hostStatusRef,
		volatile int** _deviceStatusRef)
	{
		// store indexes and device pointers got from the host instance
		arraySize = _arraySize;
		deviceArray = (item_t**)(*_deviceArrayRef);
		bufferSize = _bufferSize;
		hostStatus = *_hostStatusRef;
		deviceStatus = *_deviceStatusRef;
		index = 0;
	}

	__device__ __forceinline__ item_t* operator[](int _arrayId)
	{
		return deviceArray[_arrayId];
	}

	__device__ void print()
	{
		printf("inputDispatcherDevice\n");
		printf("    buffer size: %d  at %p\n", bufferSize, deviceArray);
		printf("    array size : %d  , at ...\n", arraySize);
		for (int i = 0; i < bufferSize; i++)
		{
			printf("    [%d] at       d:%p,    status h:%d  d:%d  ", i, deviceArray[i], hostStatus[i], deviceStatus[i]);
			for (int j = 0; j < 300; j++)
				printf("<%d>", deviceArray[i][j]);
			printf("\n\n");
		}
	}

	__device__ int get(int* _arrayIdRef) // return the index of next input and step index 
	{
		unsigned int _index = index; // saves index as can change in the background
		volatile int* thisDeviceStatusRef = &(deviceStatus[_index]); // read status address (must be volatile, otherwise write to cache only)
		if ((deviceStatus[_index] == statusHost) && (hostStatus[_index] == statusDevice))
		{			
			int oldStatus = atomicCAS((int*)(thisDeviceStatusRef), statusHost, statusDevice); // tries to book
			if (oldStatus == statusHost) // booking was succesful -> no other thread can change this item or its status
			{
				atomicInc((unsigned int*)&index, bufferSize - 1);// increment index
				*_arrayIdRef = _index; // passes the value
				return 0; // succesful
			}
		}
		else if ((deviceStatus[_index] == statusHost) && (hostStatus[_index] == statusStop)) // stop signal found
		{
			int oldStatus = atomicCAS((int*)(thisDeviceStatusRef), statusHost, statusDevice); // resets to resuse without send()
			if (oldStatus == statusHost) // booking was succesful -> no other thread can change this item or its status
			{
				atomicInc((unsigned int*)&index, bufferSize - 1);// increment index
				*_arrayIdRef = -1; // indicates, that stop was found
				deviceStatus[_index] = statusFree; // initate a send(), what does not happen because of stopping
				return 0; // succesful
			}
		}
		return -1; // unsuccessful read - because: statuses are not OK, or another thread get already
	}

	__device__ int send(int* _arrayIdRef) // passes the array to host for reuse
	{
		if (deviceStatus[*_arrayIdRef] != statusDevice) return -1; // unsuccessful: status is not OK
		deviceStatus[*_arrayIdRef] = statusFree;
		return 0; // succesful
	}
};



// ************************************* output, host

template <class item_t, class index_t>
class outputDispatcherHost_t
{
public:
	index_t arraySize; // amount of (item_t) data in one output array
	item_t** hostArray; // array of host pointers to the output data arrays
	item_t** deviceArray; // array of device pointers to the output data arrays
	item_t** deviceArray_d;

	int bufferSize; // how many arrays to manage
	int* hostStatus; // the status of the host side data arrays
	int* hostStatus_d;
	int* deviceStatus; // the status of the device side data arrays
	int* deviceStatus_d;

	unsigned int index; // index of actual  array
	cudaStream_t stream; // data are sent on this stream
	cudaEvent_t transferDone; // to record async memory transfer completion 

public:
	outputDispatcherHost_t(int _bufferSize, index_t _arraySize) : bufferSize(_bufferSize), arraySize(_arraySize)
	{
		cudaStreamCreate(&stream);
		cudaEventCreate(&transferDone);
		// init index
		index = 0;
		// allocate memory for array of pointers
		cudaMallocHost(&hostArray, bufferSize * sizeof(item_t*)); // allocates on the host 
		cudaMallocHost(&deviceArray, bufferSize * sizeof(item_t*)); // allocates on the host 
		cudaMalloc(&deviceArray_d, bufferSize * sizeof(item_t*)); // allocates on the device
		// allocate pinned memory for the arrays
		for (int i = 0; i < bufferSize; i++)
		{
			cudaMallocHost(&hostArray[i], arraySize * sizeof(item_t)); // allocates on the host 
			cudaMalloc(&deviceArray[i], arraySize * sizeof(item_t)); // allocates on the device
		}
		// copy device side array pointers to the device (only once must be done as pointers do not change) 
		cudaMemcpyAsync(deviceArray_d, deviceArray, bufferSize * sizeof(item_t*), cudaMemcpyHostToDevice, stream);
		cudaEventRecord(transferDone, stream);
		// 	allocate mapped memory for status array
		cudaHostAlloc(&hostStatus, _bufferSize * sizeof(int), cudaHostAllocMapped); // allocate on host side and map on device side
		cudaHostGetDevicePointer(&hostStatus_d, hostStatus, 0); // get device pointer
		cudaHostAlloc(&deviceStatus, _bufferSize * sizeof(int), cudaHostAllocMapped); // allocate on host side and map on device side
		cudaHostGetDevicePointer(&deviceStatus_d, deviceStatus, 0); // get device pointer
		// init statuses
		for (int i = 0; i < bufferSize; i++)
		{
			hostStatus[i] = statusDevice;
			deviceStatus[i] = statusHost;
		}
		// wait till pointer transfer is finished
		cudaEventSynchronize(transferDone);
	}

	~outputDispatcherHost_t()
	{
		for (int i = 0; i < bufferSize; i++)
			cudaFreeHost(hostArray[i]);
		cudaFreeHost(hostArray);
		cudaFreeHost(deviceArray);
		cudaFreeHost(hostStatus);
		cudaFreeHost(deviceStatus);
		cudaEventDestroy(transferDone);
		cudaStreamDestroy(stream);
	}

	item_t* operator[](int _arrayId)
	{
		return hostArray[_arrayId];
	}

	void print()
	{
		printf("outputDispatcherHost\n");
		printf("    buffer size: %d  at %p\n", bufferSize, hostArray);
		printf("    array size : %d  , at ...\n", arraySize);
		for (int i = 0; i < bufferSize; i++)
		{
			printf("    [%d] at h:%p  d:%p,    status h:%d  d:%d  ", i, hostArray[i], deviceArray[i], hostStatus[i], deviceStatus[i]);
			for (int j = 0; j < 300; j++)
				printf("<%d>", hostArray[i][j]);
			printf("\n\n");
		}
	}

	int send() // send actual if possible, step index
	{
		if (deviceStatus[index] == statusHost)
		{
			hostStatus[index] = statusDevice; // device can reuse
			return 0; // succesful
		}
		return -1; // unsuccesful
	}

	int get(int* _arrayIdRef) // get the index of actual
	{
		if ((hostStatus[index] == statusDevice) && (deviceStatus[index] == statusFree))
		{
			cudaMemcpyAsync(hostArray[index], deviceArray[index], arraySize * sizeof(item_t), cudaMemcpyDeviceToHost, stream);
			cudaEventRecord(transferDone, stream);
			cudaEventSynchronize(transferDone); // wait till copy is finished
			hostStatus[index] = statusHost; // host can read
			deviceStatus[index] = statusHost; // device can get
			*_arrayIdRef = index; // passes the value
			index = (index == bufferSize - 1) ? 0 : index + 1; // step to next array
			return 0; // succesful
		}
		return -1; // unsuccesful: not sent by device yet
	}
};


// ************************************* output, device

template <class item_t, class index_t>
class outputDispatcherDevice_t
{
public:
	index_t arraySize; // amount of (item_t) data in one output array
	item_t** deviceArray;

	int bufferSize; // how many arrays to manage
	volatile int* hostStatus;
	volatile int* deviceStatus;

	volatile unsigned int index; // index of actual  array

public:
	__device__ void setup(
		size_t _arraySize,
		void*** _deviceArrayRef,
		int _bufferSize,
		volatile int** _hostStatusRef,
		volatile int** _deviceStatusRef)
	{
		// store indexes and device pointers got from the host instance
		arraySize = _arraySize;
		deviceArray = (item_t**)(*_deviceArrayRef);
		bufferSize = _bufferSize;
		hostStatus = *_hostStatusRef;
		deviceStatus = *_deviceStatusRef;
		index = 0;
	}
	
	__device__ __forceinline__ item_t* operator[](int _arrayId)
	{
		return deviceArray[_arrayId];
	}

	__device__ void print()
	{
		printf("outputDispatcherDevice\n");
		printf("    buffer size: %d  at %p\n", bufferSize, deviceArray);
		printf("    array size : %d  , at ...\n", arraySize);
		for (int i = 0; i < bufferSize; i++)
		{
			printf("    [%d] at       d:%p,    status h:%d  d:%d  ", i, deviceArray[i], hostStatus[i], deviceStatus[i]);
			for (int j = 0; j < 300; j++)
				printf("<%d>", deviceArray[i][j]);
			printf("\n\n");
		}
	}

	__device__ int get(int* _arrayIdRef) // return the next output and step index 
	{
		unsigned int _index = index; // saves index as can change in the background
		volatile int* thisDeviceStatusRef = &(deviceStatus[_index]); // read status address (must be volatile, otherwise write to cache only)
		if ((deviceStatus[_index] == statusHost) && (hostStatus[_index] == statusDevice))
		{
			int oldStatus = atomicCAS(const_cast<int*>(thisDeviceStatusRef), statusHost, statusDevice); // tries to book
			if (oldStatus == statusHost) // booking was succesful -> no other thread can change this item or its status
			{
				atomicInc((unsigned int*)&index, bufferSize - 1);// increment index
				*_arrayIdRef = _index; // passes the value
				return 0; // succesful
			}
		}
		return -1; // unsuccessful read - because: statuses are not OK, or another thread get already
	}

	__device__ int send(int* _arrayIdRef) // passes the array to host for reuse
	{

		if (deviceStatus[*_arrayIdRef] != statusDevice) return -1; // unsuccessful: status is not OK
		deviceStatus[*_arrayIdRef] = statusFree;
		return 0; // succesful

	}
};






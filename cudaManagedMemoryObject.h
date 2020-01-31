
#pragma once



/*

managed memory base class
-------------------------
overloads new and delete to managed memory allocation model
inherit all managed memory classes from this class and construct/destroy using new/delete

by my experience managed memory is NOT useful for communication during kernel run
data migration (if necessary) happens BEFORE and AFTER kernel call 

*/

class managedMemoryObject_t {
public:
	void* operator new(size_t len) {
		void* object;
		cudaMallocManaged(&object, len);
		cudaDeviceSynchronize();
		return object;
	}

	void operator delete(void* object) {
		cudaDeviceSynchronize();
		cudaFree(object);
	}
};









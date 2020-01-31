

// checking CUDA runtime API results
// only in debug
// example: CudaCheckError( cudaMallocHost((void**)&ptr, bytes) );


#pragma once

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (cudaSuccess != err)
	{
		fprintf(stderr, "failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}


inline void __cudaCheckError(const char* file, const int line)
{
#if defined(DEBUG) || defined(_DEBUG)
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "failed with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

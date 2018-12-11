//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

const int BLOCK_SIZE = 1024;

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative off of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Comyyne the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

*/



__global__ void pred(unsigned int* pred, const unsigned int* d_in, size_t numElems,int yyt) {
	
	int global_id = threadIdx.x + blockDim.x*blockIdx.x;
	if (global_id >= numElems) return;
	unsigned int yyn = ((d_in[global_id] >> yyt) & 1u);
	pred[global_id] = yyn;
}


__global__ void scan_kernel(unsigned int* d_out, const unsigned int* d_in, size_t input_size, unsigned int* blockSums) {

	extern __shared__ unsigned int data[];
	

	int off = 1;
	int abs_start = 2*blockDim.x*blockIdx.x;
	
	data[2 * threadIdx.x] =(abs_start+2*threadIdx.x)<input_size? d_in[abs_start+2 * threadIdx.x]:0;
	data[2 * threadIdx.x+1] = (abs_start + 2 * threadIdx.x+1)<input_size ? d_in[abs_start+2 * threadIdx.x+1]:0;

	for (int a = (2 * blockDim.x) >>1; a>0; a>>=1) {
		__syncthreads();
		
		if (threadIdx.x < a) {
			int xx = off*(2 * threadIdx.x + 1) - 1;
			int yy = off*(2 * threadIdx.x + 2) - 1;
			
			data[yy] += data[xx];
		}
		off <<= 1;
	}
	if (threadIdx.x == 0)data[2*blockDim.x - 1] = 0;

	for (int a = 1; a < 2 * blockDim.x; a<<=1) {
		off >>= 1;
		__syncthreads();
		if (threadIdx.x < a) {
			int xx = off*(2 * threadIdx.x + 1) - 1;
			int yy = off*(2 * threadIdx.x + 2) - 1;
			unsigned int t = data[xx];
			data[xx] = data[yy];
			data[yy] += t;
		}
	}
	
	__syncthreads();
	
	if (abs_start + 2 * threadIdx.x < input_size) {
		d_out[abs_start + 2 * threadIdx.x] = data[2 * threadIdx.x];
	}
	if (abs_start + 2 * threadIdx.x+1 < input_size) {
		d_out[abs_start + 2 * threadIdx.x+1] = data[2 * threadIdx.x+1];
	}

	if (threadIdx.x == 0) {
		blockSums[blockIdx.x] = data[blockDim.x * 2 - 1];
		if(abs_start + blockDim.x * 2 - 1<input_size)blockSums[blockIdx.x]+=d_in[abs_start + blockDim.x * 2 - 1];
	}
}

__global__ void adjust(unsigned int* q, unsigned int* incr, size_t input_size){
	int pos = blockIdx.x * blockDim.x*2 + threadIdx.x * 2 + 1;
	if (pos< input_size)
	{
		q[pos] += incr[blockIdx.x];
		q[pos-1] += incr[blockIdx.x];
	}
	else if (pos-1 < input_size)
	{
		q[pos-1] += incr[blockIdx.x];
	}
}

__global__ void negatepred(unsigned int* pred, size_t input_size) {
	
	int pos = blockDim.x*blockIdx.x + threadIdx.x;
	if (pos >= input_size)return;
	pred[pos] = pred[pos] ? 0 : 1;
}

__global__ void moveElements(unsigned int* d_out, const unsigned int* d_in, const unsigned int* d_histo, 
								const unsigned int* d_pred,const unsigned int* d_scan_true, const unsigned int* d_scan_false, size_t input_size) {
	
	int pos = blockDim.x*blockIdx.x + threadIdx.x;
	if (pos >= input_size)return;
	//calculate new index of element at position pos
	int newindex;	
	if (d_pred[pos])newindex = d_histo[0] + d_scan_false[pos];
	else newindex = d_histo[1] + d_scan_true[pos];
	if (newindex >= input_size) return;
	d_out[newindex] = d_in[pos];
}


unsigned int scanKernel(unsigned int* d_scan, unsigned int* d_pred, size_t numElems) {
	
	int num_double_blocks = ceil(1.0f*numElems / (2*BLOCK_SIZE));
	unsigned int* d_blocksums;
	checkCudaErrors(cudaMalloc(&d_blocksums, num_double_blocks * sizeof(unsigned int)));
	scan_kernel << <num_double_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE*sizeof(unsigned int) >> > (d_scan, d_pred, numElems, d_blocksums);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	unsigned int finalSum;
	//Scan of the blocksums array
	if (num_double_blocks > 1) {
		unsigned int* d_scan_temp;
		checkCudaErrors(cudaMalloc(&d_scan_temp, num_double_blocks * sizeof(unsigned int)));
		finalSum=scanKernel(d_scan_temp, d_blocksums, num_double_blocks);
		adjust << <num_double_blocks, BLOCK_SIZE >> > (d_scan, d_scan_temp, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaFree(d_scan_temp));
	}
	else {
		
		checkCudaErrors(cudaMemcpy(&finalSum, d_blocksums, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_blocksums));
	}
	
	return finalSum;

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t numElems)
{ 
  
	int num_blocks = ceil(1.0f*numElems / BLOCK_SIZE);
	
	unsigned int h_histo[2];
	h_histo[0] = 0;

	unsigned int* d_histo;
	unsigned int* d_pred;
	unsigned int* d_scan_true;
	unsigned int* d_scan_false;
	
	checkCudaErrors(cudaMalloc(&d_histo, 2 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_pred, numElems*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_scan_true, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_scan_false, numElems * sizeof(unsigned int)));

	
	for (size_t i = 0; i < 32; i++) {

		//compute the predicate
		if (i % 2 == 0)pred << <num_blocks, BLOCK_SIZE >> > (d_pred, d_inputVals, numElems, i);
		else pred << <num_blocks, BLOCK_SIZE >> > (d_pred, d_outputVals, numElems, i);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		
	
		
		//blelloch scan
		unsigned int number_trues=scanKernel(d_scan_true, d_pred, numElems);

		//Flip yyts
		negatepred << <num_blocks, BLOCK_SIZE >> > (d_pred, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//Compute from negatives
		unsigned int number_falses=scanKernel(d_scan_false, d_pred, numElems);

		h_histo[1] = number_falses;
		checkCudaErrors(cudaMemcpy(d_histo, h_histo, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));

		//Moving elements and indices
		if (i % 2 == 0) {
			moveElements << <num_blocks, BLOCK_SIZE >> > (d_outputVals, d_inputVals, d_histo, d_pred, d_scan_true, d_scan_false, numElems);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			moveElements << <num_blocks, BLOCK_SIZE >> > (d_outputPos, d_inputPos, d_histo, d_pred, d_scan_true, d_scan_false, numElems);

		}
		else {
			moveElements << <num_blocks, BLOCK_SIZE >> > (d_inputVals, d_outputVals, d_histo, d_pred, d_scan_true, d_scan_false, numElems);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			moveElements << <num_blocks, BLOCK_SIZE >> > (d_inputPos, d_outputPos, d_histo, d_pred, d_scan_true, d_scan_false, numElems);

		}
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
	}

	
	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	
	checkCudaErrors(cudaFree(d_histo));
	checkCudaErrors(cudaFree(d_pred));
	checkCudaErrors(cudaFree(d_scan_true));
	checkCudaErrors(cudaFree(d_scan_false));

}




















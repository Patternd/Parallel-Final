//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>

const int block_x = 16;
const int block_y = 16;
const int filterWidth = 3;

__device__ void clamp(int & pos, int maxpos) {

	pos = pos > 0 ? pos : 0;
	pos = pos < (maxpos - 1) ? pos : (maxpos - 1);

}



__device__ bool mask(uchar4 val) {

	return (val.x != 255 || val.y != 255 || val.z != 255);

}


__global__ void bord_int(const uchar4* const d_sourceImg, const size_t numRowsSource, const size_t numColsSource, unsigned char* border, 														unsigned char* interior, int* 															xcoords, int* ycoords) {
	
	 __shared__ uchar4 sh_arr[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];

	//Load data in shared mem

	const int2 make_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
						blockIdx.y * blockDim.y + threadIdx.y);


	int make_1D_pos = make_2D_pos.y * numColsSource + make_2D_pos.x;
	int halfWidth = filterWidth/2; 

	int load_x_new = make_2D_pos.x - halfWidth;
	int load_y_new = make_2D_pos.y - halfWidth;

	clamp(load_x_new, numColsSource);
	clamp(load_y_new, numRowsSource);

	int load_x_original = load_x_new;
	int load_y_original = load_y_new;

	sh_arr[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_sourceImg[load_y_new*numColsSource + load_x_new];

	if (threadIdx.y >= (blockDim.y - filterWidth + 1)) {

		load_y_new = make_2D_pos.y + halfWidth;
		clamp(load_y_new, numRowsSource);
		sh_arr[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_sourceImg[load_y_new*numColsSource + 																	load_x_original];

	}	

	if (threadIdx.x >= (blockDim.x - filterWidth + 1)) {

		load_x_new = make_2D_pos.x + halfWidth;
		clamp(load_x_new, numColsSource);
		sh_arr[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = d_sourceImg[load_y_original*numColsSource + load_x_new];

	}
	if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth - 1)) {

		load_x_new = make_2D_pos.x - halfWidth + blockDim.x;
		load_y_new = make_2D_pos.y - halfWidth + blockDim.y;
		clamp(load_x_new, numColsSource);
		clamp(load_y_new, numRowsSource);
		sh_arr[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = d_sourceImg[load_y_new*numColsSource + load_x_new];

	}

	//End load data
	__syncthreads();

	if (make_2D_pos.x >= numColsSource ||
		make_2D_pos.y >= numRowsSource) return;
	
	int target_pos_x = threadIdx.x + halfWidth;
	int target_pos_y = threadIdx.y + halfWidth;
	if (!mask(sh_arr[target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x]))return; //neither interior or border



	if (!mask(sh_arr[target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x-1]) || !mask(sh_arr[target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x+1]) ||
			!mask(sh_arr[(target_pos_y-1)*(blockDim.x + filterWidth - 1) + target_pos_x]) || !mask(sh_arr[(target_pos_y+1)*(blockDim.x + filterWidth - 1) + 																			target_pos_x])) {

		border[make_1D_pos] = 1;
		xcoords[make_1D_pos] = make_2D_pos.x;
		ycoords[make_1D_pos] = make_2D_pos.y;
		return;

	}
	
	interior[make_1D_pos] = 1;

}


__device__ float _min(float a, float b) {

	if (a < 0)a = 999999;
	if (b < 0)b = 999999;
	return (a < b) ? a : b;

}

__device__ float _max(float a, float b) {

	return a > b ? a : b;

}

__global__ void minmax_reduce(int* d_out, const int* d_in, int input_size, bool isMin) {

	extern __shared__ float sdata[];

	int t_id = threadIdx.x;
	int global_id = t_id + blockDim.x*blockIdx.x;

	if (global_id >= input_size) { 
		sdata[t_id] = d_in[0]; 
	} 

	else sdata[t_id] = d_in[global_id];

	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {

		if (t_id < s) sdata[t_id] = isMin ? _min(sdata[t_id], sdata[t_id + s]) : _max(sdata[t_id], sdata[t_id + s]);

		__syncthreads();

	}

	if (t_id == 0) {

		d_out[blockIdx.x] = sdata[0];

	}
}

int reduce(const int* const d_in, int input_size, bool isMin) {
	int threads = block_x*block_y;
	int* d_current_in = NULL;
	int size = input_size;
	int blocks = ceil(1.0f*size / threads);
	while (true) {

		//allocate memory for intermediate results
		int* d_out;
		checkCudaErrors(cudaMalloc(&d_out, blocks * sizeof(int)));

		//call reduce kernel
		if (d_current_in == NULL) minmax_reduce << <blocks, threads, threads * sizeof(int) >> > (d_out, d_in, size, isMin);
		else minmax_reduce << <blocks, threads, threads * sizeof(int) >> > (d_out, d_current_in, size, isMin);;
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//free last intermediate result
		if (d_current_in != NULL) checkCudaErrors(cudaFree(d_current_in));

		if (blocks == 1) {
			
			int h_out;
			checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
			return h_out;
		}
		size = blocks;
		blocks = ceil(1.0f*size / threads);
		if (blocks == 0)blocks++;
		d_current_in = d_out;
	}

}

//This function takes an image in uchar4 and splits it into three different colored images
__global__
void separate_channels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	float* const red_channel,
	float* const green_channel,
	float* const blue_channel)
{

	const int2 make_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (make_2D_pos.x >= numCols ||
		make_2D_pos.y >= numRows)
	{
		return;
	}
	int make_1D_pos = make_2D_pos.y * numCols + make_2D_pos.x;
	red_channel[make_1D_pos] = (float)inputImageRGBA[make_1D_pos].x;
	green_channel[make_1D_pos] = (float)inputImageRGBA[make_1D_pos].y;
	blue_channel[make_1D_pos] = (float)inputImageRGBA[make_1D_pos].z;
}

__global__
void jacobi(float* const d_original_in,float* const d_in, float* const d_source_in, 					   unsigned char *border, unsigned char *interior,  					   float* d_out, int minx,int miny, 				           int numRowsSource,int numColsSource) {

	__shared__ float sh_arr_source[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];
	__shared__ float sh_arr_target[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];
	__shared__ float sh_arr_target_original[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];
	__shared__ unsigned char sh_interior[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];
	__shared__ unsigned char sh_border[(block_x + filterWidth - 1)*(block_y + filterWidth - 1)];
	//lopad data in shared mem

	const int2 make_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x+minx,
		blockIdx.y * blockDim.y + threadIdx.y+miny);


	int make_1D_pos = make_2D_pos.y * numColsSource + make_2D_pos.x;
	int halfWidth = filterWidth / 2;

	int load_x_new = make_2D_pos.x - halfWidth;
	int load_y_new = make_2D_pos.y - halfWidth;

	clamp(load_x_new, numColsSource);
	clamp(load_y_new, numRowsSource);

	int load_x_original = load_x_new;
	int load_y_original = load_y_new;

	sh_arr_source[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_source_in[load_y_new*numColsSource + load_x_new];
	sh_arr_target[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_in[load_y_new*numColsSource + load_x_new];
	sh_arr_target_original[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_original_in[load_y_new*numColsSource + load_x_new];
	sh_interior[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = interior[load_y_new*numColsSource + load_x_new];
	sh_border[threadIdx.y*(blockDim.x + filterWidth - 1) + threadIdx.x] = border[load_y_new*numColsSource + load_x_new];

	if (threadIdx.y >= (blockDim.y - filterWidth + 1)) {
		load_y_new = make_2D_pos.y + halfWidth;
		clamp(load_y_new, numRowsSource);
		sh_arr_source[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_source_in[load_y_new*numColsSource + load_x_original];
		sh_arr_target[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_in[load_y_new*numColsSource + load_x_original];
		sh_arr_target_original[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = d_original_in[load_y_new*numColsSource + load_x_original];
		sh_interior[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = interior[load_y_new*numColsSource + load_x_original];
		sh_border[(threadIdx.y + filterWidth - 1)*(blockDim.x + filterWidth - 1) + threadIdx.x] = border[load_y_new*numColsSource + load_x_original];
	}
	
	if (threadIdx.x >= (blockDim.x - filterWidth + 1)) {

		load_x_new = make_2D_pos.x + halfWidth;
		clamp(load_x_new, numColsSource);

		sh_arr_source[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = d_source_in[load_y_original*numColsSource + load_x_new];
		sh_arr_target[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = d_in[load_y_original*numColsSource + load_x_new];
		sh_arr_target_original[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = d_original_in[load_y_original*numColsSource + 																					load_x_new];
		sh_interior[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = interior[load_y_original*numColsSource + load_x_new];
		sh_border[(threadIdx.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + filterWidth - 1] = border[load_y_original*numColsSource + load_x_new];
	}

	if (threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth - 1)) {

		load_x_new = make_2D_pos.x - halfWidth + blockDim.x;
		load_y_new = make_2D_pos.y - halfWidth + blockDim.y;
		clamp(load_x_new, numColsSource);
		clamp(load_y_new, numRowsSource);

		sh_arr_source[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = d_source_in[load_y_new*numColsSource + load_x_new];
		sh_arr_target[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = d_in[load_y_new*numColsSource + load_x_new];
		sh_arr_target_original[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = d_original_in[load_y_new*numColsSource + 																				load_x_new];
		sh_interior[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = interior[load_y_new*numColsSource + load_x_new];
		sh_border[(threadIdx.y + blockDim.y)*(blockDim.x + filterWidth - 1) + threadIdx.x + blockDim.x] = border[load_y_new*numColsSource + load_x_new];
	}

	//end load data
	__syncthreads();

	
	if (make_2D_pos.x >= numColsSource ||
		make_2D_pos.y >= numRowsSource) return;
	
	if (interior[make_1D_pos] != 1)return; //filter out boundary points

	int target_pos_x = threadIdx.x + halfWidth;
	int target_pos_y = threadIdx.y + halfWidth;
	float valsource = sh_arr_source[target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x];

	float sum1=0.0f;
	float sum2 = 4 * valsource;

	int curpos[] = { target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x - 1,target_pos_y*(blockDim.x + filterWidth - 1) + target_pos_x + 1,
						(target_pos_y - 1)*(blockDim.x + filterWidth - 1) + target_pos_x,(target_pos_y + 1)*(blockDim.x + filterWidth - 1) + target_pos_x };

	for (int i = 0; i < 4; i++) {
		if (sh_interior[curpos[i]]) {
			sum1 += sh_arr_target[curpos[i]];
			
		}
		else if (sh_border[curpos[i]]) {
			sum1 += sh_arr_target_original[curpos[i]];
			
		}
		sum2 -= sh_arr_source[curpos[i]];
	}
	
	float newVal = (sum1 + sum2) / 4.0f;
	newVal = newVal < 0 ? 0 : newVal;
	newVal = newVal > 255 ? 255 : newVal;

	d_out[make_1D_pos] = newVal;
}

__global__
void recomb_channel(const uchar4* const d_destImg,const float* const red_channel,
	const float* const green_channel,
	const float* const blue_channel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols,unsigned char* interior)
{
	const int2 make_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int make_1D_pos = make_2D_pos.y * numCols + make_2D_pos.x;

	//doesn't access memory outside of image
	if (make_2D_pos.x >= numCols || make_2D_pos.y >= numRows)
		return;

	if (interior[make_1D_pos] != 1) {
		outputImageRGBA[make_1D_pos] = d_destImg[make_1D_pos];
		return;
	}

	unsigned char red = (unsigned char)red_channel[make_1D_pos];
	unsigned char green = (unsigned char)green_channel[make_1D_pos];
	unsigned char blue = (unsigned char)blue_channel[make_1D_pos];

	//alpha is 255
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);
	
	outputImageRGBA[make_1D_pos] = outputPixel;
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{


	const dim3 blockSize(block_x, block_y);
	const dim3 gridSize(ceil(1.0f*numColsSource / blockSize.x), ceil(1.0f*numRowsSource / blockSize.y));

	
	

	uchar4* d_sourceImg;
	unsigned char *border,*interior;
	int *xcoords, *ycoords; //for bounding box computation

	checkCudaErrors(cudaMalloc(&d_sourceImg, numRowsSource*numColsSource * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(&border, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&interior, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&xcoords, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMalloc(&ycoords, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMemset(border, 0, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(interior, 0, numRowsSource*numColsSource * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(xcoords, -1, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMemset(ycoords, -1, numRowsSource*numColsSource * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyHostToDevice));

	bord_int << <gridSize, blockSize >> > (d_sourceImg, numRowsSource, numColsSource, border, interior, xcoords, ycoords);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	
	int minx = reduce(xcoords, numRowsSource*numColsSource, true);
	int maxx = reduce(xcoords, numRowsSource*numColsSource, false);
	int miny = reduce(ycoords, numRowsSource*numColsSource, true);
	int maxy = reduce(ycoords, numRowsSource*numColsSource, false);

	int size_x = maxx - minx+1;
	int size_y = maxy - miny + 1;

	checkCudaErrors(cudaFree(xcoords));
	checkCudaErrors(cudaFree(ycoords));

	
	

	uchar4* d_destImg;

	checkCudaErrors(cudaMalloc(&d_destImg, numRowsSource*numColsSource * sizeof(uchar4)));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyHostToDevice));

	float *d_buffer_red_1, *d_buffer_red_2;
	float *d_buffer_green_1, *d_buffer_green_2;
	float *d_buffer_blue_1, *d_buffer_blue_2;
	float *d_red, *d_green, *d_blue;

	checkCudaErrors(cudaMalloc(&d_buffer_red_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_red_2, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_green_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_green_2, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_blue_1, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_buffer_blue_2, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_red, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_green, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_blue, numRowsSource*numColsSource * sizeof(float)));
	
	separate_channels << <gridSize, blockSize >> > (d_destImg, numRowsSource, numColsSource, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	
	float *d_red_source, *d_green_source, *d_blue_source;

	checkCudaErrors(cudaMalloc(&d_red_source, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_green_source, numRowsSource*numColsSource * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_blue_source, numRowsSource*numColsSource * sizeof(float)));
	separate_channels << <gridSize, blockSize >> > (d_sourceImg, numRowsSource, numColsSource, d_red_source, d_green_source, d_blue_source);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	cudaStream_t s1, s2, s3;
	cudaStreamCreate(&s1); cudaStreamCreate(&s2); cudaStreamCreate(&s3);

	checkCudaErrors(cudaMemcpyAsync(d_buffer_red_1, d_red_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice,s1));
	checkCudaErrors(cudaMemcpyAsync(d_buffer_green_1, d_green_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice,s2));
	checkCudaErrors(cudaMemcpyAsync(d_buffer_blue_1, d_blue_source, numRowsSource*numColsSource * sizeof(float), cudaMemcpyDeviceToDevice,s3));


	
	

	const dim3 gridSizeNew(ceil(1.0f*size_x / blockSize.x), ceil(1.0f*size_y / blockSize.y));
	
	for (size_t i = 0; i < 800; i++) {

		if (i % 2 == 0) {

			//source is buffer 1
			jacobi << <blockSize, gridSizeNew,0,s1 >> > (d_red,d_buffer_red_1,d_red_source,border,interior,d_buffer_red_2,minx,miny,numRowsSource,numColsSource);
			jacobi << <blockSize, gridSizeNew,0,s2 >> > (d_green, d_buffer_green_1, d_green_source, border, interior, d_buffer_green_2, minx, miny, numRowsSource, 																					numColsSource);
			jacobi << <blockSize, gridSizeNew,0,s3 >> > (d_blue, d_buffer_blue_1, d_blue_source, border, interior, d_buffer_blue_2, minx, miny, numRowsSource, 																					numColsSource);
		}

		else {
			//source is buffer 2
			jacobi << <blockSize, gridSizeNew,0,s1 >> > (d_red, d_buffer_red_2, d_red_source, border, interior, d_buffer_red_1, minx, miny, numRowsSource, 																					numColsSource);
			jacobi << <blockSize, gridSizeNew,0,s2 >> > (d_green, d_buffer_green_2, d_green_source, border, interior, d_buffer_green_1, minx, miny, numRowsSource, 																					numColsSource);
			jacobi << <blockSize, gridSizeNew,0,s3 >> > (d_blue, d_buffer_blue_2, d_blue_source, border, interior, d_buffer_blue_1, minx, miny, numRowsSource, 																				numColsSource);		
		}
		
	}

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	cudaStreamDestroy(s1); cudaStreamDestroy(s2); cudaStreamDestroy(s3);

	uchar4* d_blendedImg;

	checkCudaErrors(cudaMalloc(&d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4)));
	recomb_channel << <blockSize, gridSize >> > (d_destImg,d_buffer_red_1, d_buffer_green_1, d_buffer_blue_1, d_blendedImg, numRowsSource, numColsSource,interior);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_sourceImg));
	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(border));
	checkCudaErrors(cudaFree(interior));
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_red_source));
	checkCudaErrors(cudaFree(d_green_source));
	checkCudaErrors(cudaFree(d_blue_source));
	checkCudaErrors(cudaFree(d_buffer_red_1));
	checkCudaErrors(cudaFree(d_buffer_red_2));
	checkCudaErrors(cudaFree(d_buffer_green_1));
	checkCudaErrors(cudaFree(d_buffer_green_2));
	checkCudaErrors(cudaFree(d_buffer_blue_1));
	checkCudaErrors(cudaFree(d_buffer_blue_2));
}

	
	


				

     


	

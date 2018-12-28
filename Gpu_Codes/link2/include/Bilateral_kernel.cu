#include "Bilateral_kernel.h"
//#include <cuda_profiler_api.h>
#include "Timer.h"
#include <cuda.h>

#define BLOCK_SIZE 16
#define ABS(n) ((n) < 0 ? -(n) : (n))

// Constant
__constant__ float CUDA_Gaussian2D[10202];
__constant__ float CUDA_Gauss1D[256];

// Textures
texture<unsigned char> CUDA_Data;
texture<unsigned char, cudaTextureType2D> CUDA_Frame;
texture<float> CUDA_Kernel;
texture<float> CUDA_Gaussian1D;

static inline void _safe_cuda_call(cudaError e, const char *msg,
								const char *file, const int line){
	if(e!=cudaSuccess){
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",
				msg, file, line, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call, msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)

// DEVICE FUNCTIONS
__device__ inline float gaussian_1d(const int x, const float sigma){
	float v1=2*sigma*sigma;
	float v2=sqrt(2*3.1415926)*sigma;
	float e=-(x*x)/v1;
	return expf(e)/v2;
}

__device__ inline float gaussian_2d(const int x, const int y, const float sigma){
	float v=2*sigma*sigma;
	float e=-((x*x)+(y*y))/v;
	return expf(e)/(3.1415926*v);
}

__device__ inline bool block(const int x, const int y){
	if( x>=0 && x<BLOCK_SIZE && y>=0 && y<BLOCK_SIZE )
		return true;
	return false;
}

__global__ void mask_calc_kernel(float *g_1d, float *g_2d, const int radius, const float s,
							const float r){
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int size=2*radius+1;
	if((x>=size)||(y>=size))
		return;
	
	//Calculate mask gaussian 2d
	g_2d[x*size+y]=gaussian_2d(x, y, s);

	//Calculate mask gaussian 1d
	const int w=threadIdx.y*16+threadIdx.x;
	if(w<256)
		g_1d[w]=gaussian_1d(w, r);

}

__device__ int global2D(unsigned char *GPU_input_global, int x, int y, int width, int height){
	if( x < 0 || x >= width || y < 0 || y >= width){
//		printf("boundary error \n"  );
		return 0;
	} else {
//		printf("%d \n", (u nsigned int)GPU_input_global[y*width + x]);
		return (unsigned int)GPU_input_global[y*width + x];
	}
}

__device__ int shared2D(unsigned int img_part[32+1][32*3+1], int x, int y, int radius){
	return img_part[y+(radius+1)][x+(radius+1)*3];
}

__device__ float kernel(const int x, const int y, float *g2d, float *g1d, const int radius, const int width,
														const int height, unsigned char *GPU_input_global){


		float value=0.0, k=0.0;
		int  pixel = global2D(GPU_input_global, x, y, width, height);


		int pos_value=0, pos_k=0;

		for(int i=-radius; i<=radius; i++){
			for(int channel_pos=-radius*3; channel_pos<=radius*3; channel_pos+=3){
				value+=global2D(GPU_input_global, x-channel_pos, y+i, width, height)*g2d[pos_value++]*g1d[ABS(pixel-global2D(GPU_input_global, x-channel_pos, y+i, width, height))];
			}
			for(int channel_pos=-radius*3; channel_pos<=radius*3; channel_pos+=3){
				k+=g2d[pos_k++]*g1d[ABS(pixel-global2D(GPU_input_global, x-channel_pos, y+i, width, height))];
			}
		}

	return value/k;
}

__global__ void bilateral_filter(unsigned char *GPU_input_global ,unsigned char *GPU_output_global, const int width, const int height, const int radius, const float *mask, const float *gauss, const int block_size){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int shared_height = block_size+2*(radius+1);
	const int shared_width = (block_size+2*(radius+1)) *3;

	// pixels at edges need radius+1 pixels at left/right/up/bottom
	__shared__ unsigned int img_part[16+2*(7+1)+1][(16+2*(7+1))*3+1];

	int copy_size_x = shared_width/block_size;
	int copy_size_y = shared_height/block_size;

	int copy_start_y = blockIdx.y * blockDim.y - (radius+1);
	int copy_start_x = blockIdx.x * blockDim.x - (radius+1)*3;

	for(int shared_x=threadIdx.x*copy_size_x; shared_x<(threadIdx.x+1)*copy_size_x; shared_x++){
		for(int shared_y=threadIdx.y*copy_size_y; shared_y<(threadIdx.y+1)*copy_size_y; shared_y++){
			img_part[shared_y][shared_x] = global2D(GPU_input_global, copy_start_x+shared_x, copy_start_y+shared_y, width, height);
		}
	}
	__syncthreads();


//	printf("x: %d y: %d blockIdx.x: %d blockIdx.y: %d \n", x,y,blockIdx.x, blockIdx.y);
	const int size=2*radius+1;
	extern __shared__ float g2d[];
	if( threadIdx.x<size && threadIdx.y<size ){
		g2d[threadIdx.y*size+threadIdx.x]=mask[threadIdx.y*size+threadIdx.x];
	}

	__shared__ float gaussian[256];
	const int w=threadIdx.y*16+threadIdx.x;
	if(w<256) 	gaussian[w]=gauss[w];

	__syncthreads();

	if((x>=width)||(y>=height))
 		return;

	float result=0.0;
	result=kernel(x, y, g2d, gaussian, radius, width, height, GPU_input_global);
	GPU_output_global[y*width+x]=(unsigned char) result;

}

CUDABilateralFilter::CUDABilateralFilter(const int r, const float sigma_s, const float sigma_r) :
radius(r), ss(sigma_s), sr(sigma_r) {}

void CUDABilateralFilter::apply(const Mat &input, Mat &output){
	const int width=input.cols*3, height=input.rows; // 3 columns for RGB
	const dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid;
	unsigned char *GPU_input_global, *GPU_output_global;

	float *g_1d, *g_2d;
	size_t gpu_image_size=width*height*sizeof(unsigned char);

//	printf("%d\n", input.data[10][10]);
//	printf("%d \n", input.rows);

	SAFE_CALL(cudaMalloc((unsigned char**)&GPU_input_global, gpu_image_size), "CUDA MALLOC Input");
	SAFE_CALL(cudaMalloc((unsigned char**)&GPU_output_global, gpu_image_size), "CUDA MALLOC Output");

	Timer t;
	float tenTime = 0.0;

	for(int i=0; i<10; i++){

		t.start();
		SAFE_CALL(cudaMemcpy(GPU_input_global, input.data, gpu_image_size, cudaMemcpyHostToDevice), "CUDA MEMCPY HOST TO DEVICE");

		Grid.x=(width+Block.x-1)/Block.x;
		Grid.y=(height+Block.y-1)/Block.y;

		// Gaussian 2D
		const int size=(2*radius+1);
		const size_t dim=size*size*sizeof(float);
		SAFE_CALL(cudaMalloc((void**)&g_2d, size*size*sizeof(float)), "CUDA MALLOC Mask");

		//Gaussian 1D
		SAFE_CALL(cudaMalloc((void**)&g_1d, 256*sizeof(float)), "CUDA MALLOC Mask");

		dim3 mask;
		mask.x=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
		mask.y=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
		mask_calc_kernel<<<mask, Block>>>(g_1d, g_2d, radius, ss, sr);
		SAFE_CALL(cudaDeviceSynchronize(), "CUDA DEVICE SYNCHRONIZE Mask");


		bilateral_filter<<<Grid, Block, dim>>>(GPU_input_global, GPU_output_global, width, height, radius, g_2d, g_1d, BLOCK_SIZE);
		SAFE_CALL(cudaDeviceSynchronize(), "CUDA DEVICE SYNCHRONIZE");
	
		SAFE_CALL(cudaMemcpy(output.data,GPU_output_global, gpu_image_size, cudaMemcpyDeviceToHost), "CUDA MEMCPY DEVICE TO HOST");
		t.stop();
//		t.printTime();
		printf("%d.loop: %f\n", i, t.getTime());
		tenTime += t.getTime();
	}
	printf("Average: %f\n", tenTime/10);

	SAFE_CALL(cudaFree(g_2d), "CUDA FREE");
	SAFE_CALL(cudaFree(g_1d), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_input_global), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_output_global), "CUDA FREE");
}


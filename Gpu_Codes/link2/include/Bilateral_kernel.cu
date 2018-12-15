#include "Bilateral_kernel.h"
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

__device__ float kernel(int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y);
	int pos_value=0, pos_k=0;
	
	for(int i=-5; i<6; i++){
		for(int channel_pos=-15; channel_pos<=15; channel_pos+=3){
			value+=tex2D(CUDA_Frame, x-channel_pos, y+i)*g2d[pos_value++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-channel_pos, y+i))];
		}
		for(int channel_pos=-15; channel_pos<=15; channel_pos+=3){
			k+=g2d[pos_k++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-channel_pos, y+i))];
		}
	}

	return value/k;
}

__global__ void bilateral_kernel(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *mask, const float *gauss){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

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
	result=kernel(x, y, g2d, gaussian);
	out[y*pitch+x]=(unsigned char) result;
}

CUDABilateralFilter::CUDABilateralFilter(const int r, const float sigma_s, const float sigma_r) :
radius(r), ss(sigma_s), sr(sigma_r) {}

void CUDABilateralFilter::apply(const Mat &input, Mat &output){
	const int width=input.cols*3, height=input.rows;
	const dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid;
	unsigned char *GPU_input, *GPU_output;
	float *g_1d, *g_2d;
	size_t gpu_image_pitch=0;

	SAFE_CALL(cudaMallocPitch<unsigned char>(&GPU_input, &gpu_image_pitch, width, height), "CUDA MALLOC PITCH");
	SAFE_CALL(cudaMallocPitch<unsigned char>(&GPU_output, &gpu_image_pitch, width, height), "CUDA MALLOC PITCH");

	// değişebilir
	SAFE_CALL(cudaBindTexture2D(NULL, CUDA_Frame, GPU_input, width, height, gpu_image_pitch), "CUDA BIND TEXTURE");
	CUDA_Frame.addressMode[0] = CUDA_Frame.addressMode[1] = cudaAddressModeBorder;

	SAFE_CALL(cudaMemcpy2D(GPU_input, gpu_image_pitch, input.data, width, width, height, cudaMemcpyHostToDevice), "CUDA MEMCPY 2D HOST TO DEVICE");

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

	Timer t;

	t.start();
	bilateral_kernel<<<Grid, Block, dim>>>(GPU_output, width, height, gpu_image_pitch, radius, g_2d, g_1d);
	t.stop();
	SAFE_CALL(cudaDeviceSynchronize(), "CUDA DEVICE SYNCHRONIZE");
	t.printTime();

	SAFE_CALL(cudaMemcpy2D(output.data, width, GPU_output, gpu_image_pitch, width, height, cudaMemcpyDeviceToHost), "CUDA MEMCPY2D DEVICE TO HOST");

	SAFE_CALL(cudaUnbindTexture(CUDA_Frame), "CUDA UNBIND TEXTURE");

	SAFE_CALL(cudaFree(g_2d), "CUDA FREE");
	SAFE_CALL(cudaFree(g_1d), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_input), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_output), "CUDA FREE");
}


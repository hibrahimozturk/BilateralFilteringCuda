#include "Bilateral_kernel.h"
#include "Timer.h"

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
	
	//Calculo mask gaussian 2d
	g_2d[x*size+y]=gaussian_2d(x, y, s);

	//Calculo mask gaussian 1d
	const int w=threadIdx.y*16+threadIdx.x;
	if(w<256)
		g_1d[w]=gaussian_1d(w, r);

}

__device__ float kernel_r5(const int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y), pos=0;
	
	for(int i=-5; i<6; i++){
		value+=tex2D(CUDA_Frame, x-15, y+i)*g2d[pos]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-15, y+i))]
			+tex2D(CUDA_Frame, x-12, y+i)*g2d[pos+1]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-12, y+i))]
			+tex2D(CUDA_Frame, x-9, y+i)*g2d[pos+2]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+tex2D(CUDA_Frame, x-6, y+i)*g2d[pos+3]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+tex2D(CUDA_Frame, x-3, y+i)*g2d[pos+4]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+tex2D(CUDA_Frame, x, y+i)*g2d[pos+5]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+tex2D(CUDA_Frame, x+3, y+i)*g2d[pos+6]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+tex2D(CUDA_Frame, x+6, y+i)*g2d[pos+7]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+tex2D(CUDA_Frame, x+9, y+i)*g2d[pos+8]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))]
			+tex2D(CUDA_Frame, x+12, y+i)*g2d[pos+9]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+12, y+i))]
			+tex2D(CUDA_Frame, x+15, y+i)*g2d[pos+10]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+15, y+i))];
		k+=g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-15, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-12, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+12, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+15, y+i))];
	}

	return value/k;
}

__device__ float kernel_r4(const int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y), pos=0;
	
	for(int i=-4; i<5; i++){
		value+=tex2D(CUDA_Frame, x-12, y+i)*g2d[pos]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-12, y+i))]
			+tex2D(CUDA_Frame, x-9, y+i)*g2d[pos+1]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+tex2D(CUDA_Frame, x-6, y+i)*g2d[pos+2]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+tex2D(CUDA_Frame, x-3, y+i)*g2d[pos+3]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+tex2D(CUDA_Frame, x, y+i)*g2d[pos+4]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+tex2D(CUDA_Frame, x+3, y+i)*g2d[pos+5]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+tex2D(CUDA_Frame, x+6, y+i)*g2d[pos+6]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+tex2D(CUDA_Frame, x+9, y+i)*g2d[pos+7]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))]
			+tex2D(CUDA_Frame, x+12, y+i)*g2d[pos+8]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+12, y+i))];
		k+=g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-12, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+12, y+i))];
	}

	return value/k;
}

__device__ float kernel_r3(const int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y), pos=0;
	
	for(int i=-3; i<4; i++){
		value+=tex2D(CUDA_Frame, x-9, y+i)*g2d[pos]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+tex2D(CUDA_Frame, x-6, y+i)*g2d[pos+1]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+tex2D(CUDA_Frame, x-3, y+i)*g2d[pos+2]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+tex2D(CUDA_Frame, x, y+i)*g2d[pos+3]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+tex2D(CUDA_Frame, x+3, y+i)*g2d[pos+4]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+tex2D(CUDA_Frame, x+6, y+i)*g2d[pos+5]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+tex2D(CUDA_Frame, x+9, y+i)*g2d[pos+6]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))];
		k+=g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-9, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+9, y+i))];
	}

	return value/k;
}

__device__ float kernel_r2(const int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y), pos=0;
	
	for(int i=-2; i<3; i++){
		value+=tex2D(CUDA_Frame, x-6, y+i)*g2d[pos]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+tex2D(CUDA_Frame, x-3, y+i)*g2d[pos+1]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+tex2D(CUDA_Frame, x, y+i)*g2d[pos+2]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+tex2D(CUDA_Frame, x+3, y+i)*g2d[pos+3]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+tex2D(CUDA_Frame, x+6, y+i)*g2d[pos+4]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))];
		k+=g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-6, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+6, y+i))];
	}

	return value/k;
}

__device__ float kernel_r1(const int x, const int y, float *g2d, float *g1d){
	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y), pos=0;
	
	for(int i=-1; i<2; i++){
		value+=tex2D(CUDA_Frame, x-3, y+i)*g2d[pos]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+tex2D(CUDA_Frame, x, y+i)*g2d[pos+1]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+tex2D(CUDA_Frame, x+3, y+i)*g2d[pos+2]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))];
		k+=g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x-3, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x, y+i))]
			+g2d[pos++]*g1d[ABS(pixel-tex2D(CUDA_Frame, x+3, y+i))];
	}

	return value/k;
}

__global__ void bilateral_kernel_v7(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *mask, const float *gauss){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int size=2*radius+1;
	extern __shared__ float g2d[];
	if( threadIdx.x<size && threadIdx.y<size ){
		g2d[threadIdx.y*size+threadIdx.x]=mask[threadIdx.y*size+threadIdx.x];
		if( (threadIdx.x+32)<size )
			g2d[threadIdx.y*size+(threadIdx.x+32)]=mask[threadIdx.y*size+(threadIdx.x+32)];
		if( (threadIdx.y+32)<size )
			g2d[(threadIdx.y+32)*size+threadIdx.x]=mask[(threadIdx.y+32)*size+threadIdx.x];
		if( (threadIdx.x+32)<size && (threadIdx.y+32)<size )
			g2d[(threadIdx.y+32)*size+(threadIdx.x+32)]=mask[(threadIdx.y+32)*size+(threadIdx.x+32)];
	}

	__shared__ float gaussian[256];
	const int w=threadIdx.y*16+threadIdx.x;
	if(w<256) 	gaussian[w]=gauss[w];

	__syncthreads();

	if((x>=width)||(y>=height))
		return;

	float result=0.0;
	switch(radius){
		case 1: result=kernel_r1(x, y, g2d, gaussian); break;
		case 2: result=kernel_r2(x, y, g2d, gaussian); break;
		case 3: result=kernel_r3(x, y, g2d, gaussian); break;
		case 4: result=kernel_r4(x, y, g2d, gaussian); break;
		case 5: result=kernel_r5(x, y, g2d, gaussian); break;
		default: result=0.0; break;
	}
	out[y*pitch+x]=(unsigned char) result;
}

__global__ void bilateral_kernel_v6(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *mask, const float *gauss){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int size=2*radius+1;
	extern __shared__ float kernel[];
	if( threadIdx.x<size && threadIdx.y<size ){
		kernel[threadIdx.y*size+threadIdx.x]=mask[threadIdx.y*size+threadIdx.x];
		if( (threadIdx.x+32)<size )
			kernel[threadIdx.y*size+(threadIdx.x+32)]=mask[threadIdx.y*size+(threadIdx.x+32)];
		if( (threadIdx.y+32)<size )
			kernel[(threadIdx.y+32)*size+threadIdx.x]=mask[(threadIdx.y+32)*size+threadIdx.x];
		if( (threadIdx.x+32)<size && (threadIdx.y+32)<size )
			kernel[(threadIdx.y+32)*size+(threadIdx.x+32)]=mask[(threadIdx.y+32)*size+(threadIdx.x+32)];
	}

	__shared__ float gaussian[256];
	const int w=threadIdx.y*16+threadIdx.x;
	if(w<256) 	gaussian[w]=gauss[w];

	__syncthreads();

	if((x>=width)||(y>=height))
		return;

	float value=0.0, v, val, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y);
	int pos=0, index;
	
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			v=kernel[pos++];
			index=pixel-tex2D(CUDA_Frame, x+j, y+i);
			val=v*gaussian[ABS(index)];

			k+=val;
			value+=val*tex2D(CUDA_Frame, x+j, y+i);
		}

	value/=k;
	out[y*pitch+x]=(unsigned char) value;
}

__global__ void bilateral_kernel_v5(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *gauss_1D){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if((x>=width)||(y>=height))
		return;

	float value=0.0, v, val, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y);
	int pos=0, index;
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			v=CUDA_Gaussian2D[pos++];
			index=pixel-tex2D(CUDA_Frame, x+j, y+i);
			val=v*gauss_1D[ABS(index)];

			k+=val;
			value+=val*tex2D(CUDA_Frame, x+j, y+i);
		}

	out[y*pitch+x]=(unsigned char) value/k;
}

__global__ void bilateral_kernel_v4(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *mask, const float *gauss_1D){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int size=2*radius+1;
	extern __shared__ float kernel[];
	if( threadIdx.x<size && threadIdx.y<size ){
		kernel[threadIdx.y*size+threadIdx.x]=mask[threadIdx.y*size+threadIdx.x];
		if( (threadIdx.x+32)<size )
			kernel[threadIdx.y*size+(threadIdx.x+32)]=mask[threadIdx.y*size+(threadIdx.x+32)];
		if( (threadIdx.y+32)<size )
			kernel[(threadIdx.y+32)*size+threadIdx.x]=mask[(threadIdx.y+32)*size+threadIdx.x];
		if( (threadIdx.x+32)<size && (threadIdx.y+32)<size )
			kernel[(threadIdx.y+32)*size+(threadIdx.x+32)]=mask[(threadIdx.y+32)*size+(threadIdx.x+32)];
	}

	__syncthreads();

	if((x>=width)||(y>=height))
		return;

	float value=0.0, v, val, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y);

	int pos=0, index;
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			v=kernel[pos++];
			index=pixel-tex2D(CUDA_Frame, x+j, y+i);
			val=v*gauss_1D[ABS(index)];

			k+=val;
			value+=val*tex2D(CUDA_Frame, x+j, y+i);
		}

	value/=k;
	out[y*pitch+x]=(unsigned char) value;
}

__global__ void bilateral_kernel_v3(unsigned char *out, const int width, const int height,
								const size_t pitch, const int radius, const float *mask, const float *gauss_1d){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if((x>=width)||(y>=height))
		return;

	float value=0.0, k=0.0;
	int pixel=tex2D(CUDA_Frame, x, y);

	int pos=0;
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			int index=pixel-tex2D(CUDA_Frame, x+j, y+i);
			float v=mask[pos++]*gauss_1d[ABS(index)];

			k+=v;
			value+=v*tex2D(CUDA_Frame, x+j, y+i);
		}

	value/=k;
	out[y*pitch+x]=(unsigned char) value;
}

__global__ void bilateral_kernel_v18(const unsigned char *in, unsigned char *out, const int width,
							const int height, const int radius, const float *mask, const float *gauss_1D){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if((x>=width)||(y>=height))
		return;

	__shared__ unsigned char img[BLOCK_SIZE][BLOCK_SIZE];
	img[threadIdx.y][threadIdx.x]=in[y*width+x];

	__syncthreads();

	float value=0.0, k=0.0, v;
	float pixel=in[y*width+x];

	int pos=0, index;
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			if( ((y+i)>=0) && ((y+i)<height) && ((x+j)>=0) && ((x+j)<width) )
				if( block(threadIdx.x+j, threadIdx.y+i) ){
					index=pixel-img[threadIdx.y+i][threadIdx.x+j];
					v=mask[pos]*gauss_1D[ABS(index)];

					k+=v;
					value+=v*img[threadIdx.y+i][threadIdx.x+j];
				}else{
					index=pixel-in[(y+i)*width+(x+j)];
					v=mask[pos]*gauss_1D[ABS(index)];

					k+=v;
					value+=v*in[(y+i)*width+(x+j)];
				}
			pos++;
		}

	value/=k;
	out[y*width+x]=(unsigned char) value;
}

__global__ void bilateral_kernel_v2(const unsigned char *in, unsigned char *out, const int width,
							const int height, const int radius, const float *mask, const float *gauss_1D){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if((x>=width)||(y>=height))
		return;

	float value=0.0, k=0.0;
	float pixel=in[y*width+x];

	int pos=0;
	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3){
			if( ((y+i)>=0) && ((y+i)<height) && ((x+j)>=0) && ((x+j)<width) ){
				int index=pixel-in[(y+i)*width+(x+j)];
				float v=mask[pos]*gauss_1D[ABS(index)];

				k+=v;
				value+=v*in[(y+i)*width+(x+j)];
			}
			pos++;
		}

	value/=k;
	out[y*width+x]=(unsigned char) value;
}

__global__ void bilateral_kernel_v1(const unsigned char *in, unsigned char *out, const int width,
							const int height, const int radius, const float s, const float r){

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if((x>=width)||(y>=height))
		return;

	float value=0.0, k=0.0;
	float pixel=in[y*width+x];

	for(int i=-radius; i<=radius; i++)
		for(int j=-radius*3; j<=radius*3; j+=3)
			if( ((y+i)>=0) && ((y+i)<height) && ((x+j)>=0) && ((x+j)<width) ){
				int index=pixel-in[(y+i)*width+(x+j)];
				float v=gaussian_2d(j/3, i, s)*gaussian_1d(index, r);

				k+=v;
				value+=v*in[(y+i)*width+(x+j)];
			}

	value/=k;
	out[y*width+x]=(unsigned char) value;
}

CUDABilateralFilter::CUDABilateralFilter(const int r, const float sigma_s, const float sigma_r) :
radius(r), ss(sigma_s), sr(sigma_r) {}

void CUDABilateralFilter::apply(const Mat &input, Mat &output){
	const int width=input.cols*3, height=input.rows;
	const dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid;
	unsigned char *GPU_input, *GPU_output;
	float *g_1d, *g_2d;
	//size_t gpu_image_pitch=width*height*sizeof(unsigned char);
	size_t gpu_image_pitch=0;

	//SAFE_CALL(cudaMalloc((void**)&GPU_input, gpu_image_pitch), "CUDA MALLOC Input");
	//SAFE_CALL(cudaMalloc((void**)&GPU_output, gpu_image_pitch), "CUDA MALLOC Output");
	SAFE_CALL(cudaMallocPitch<unsigned char>(&GPU_input, &gpu_image_pitch, width, height), "CUDA MALLOC PITCH");
	SAFE_CALL(cudaMallocPitch<unsigned char>(&GPU_output, &gpu_image_pitch, width, height), "CUDA MALLOC PITCH");

	//SAFE_CALL(cudaBindTexture(NULL, CUDA_Data, GPU_input, gpu_image_pitch), "CUDA BIND TEXTURE");
	SAFE_CALL(cudaBindTexture2D(NULL, CUDA_Frame, GPU_input, width, height, gpu_image_pitch), "CUDA BIND TEXTURE");
	CUDA_Frame.addressMode[0] = CUDA_Frame.addressMode[1] = cudaAddressModeBorder;

	//SAFE_CALL(cudaMemcpy(GPU_input, input.data, gpu_image_pitch, cudaMemcpyHostToDevice), "CUDA MEMCPY HOST TO DEVICE");
	SAFE_CALL(cudaMemcpy2D(GPU_input, gpu_image_pitch, input.data, width, width, height, cudaMemcpyHostToDevice), "CUDA MEMCPY 2D HOST TO DEVICE");

	Grid.x=(width+Block.x-1)/Block.x;
	Grid.y=(height+Block.y-1)/Block.y;

	// Gaussian 2D
	const int size=(2*radius+1);
	const size_t dim=size*size*sizeof(float);
	SAFE_CALL(cudaMalloc((void**)&g_2d, size*size*sizeof(float)), "CUDA MALLOC Mask");
	//SAFE_CALL(cudaBindTexture(NULL, CUDA_Kernel, g_2d, size*size*sizeof(float)), "CUDA BIND TEXTURE");

	//Gaussian 1D
	SAFE_CALL(cudaMalloc((void**)&g_1d, 256*sizeof(float)), "CUDA MALLOC Mask");
	
	dim3 mask;
	mask.x=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
	mask.y=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
	mask_calc_kernel<<<mask, Block>>>(g_1d, g_2d, radius, ss, sr);
	SAFE_CALL(cudaDeviceSynchronize(), "CUDA DEVICE SYNCHRONIZE Mask");

	//SAFE_CALL(cudaMemcpyToSymbol(CUDA_Gaussian2D, g_2d, dim), "CUDA MEM CPY TO SYMBOL");
	//SAFE_CALL(cudaBindTexture(NULL, CUDA_Gaussian1D, g_1d, 256*sizeof(float)), "CUDA BIND TEXTURE");

	Timer t;
	cudaFuncSetCacheConfig(bilateral_kernel_v7, cudaFuncCachePreferL1);

	t.start();
	bilateral_kernel_v7<<<Grid, Block, dim>>>(GPU_output, width, height, gpu_image_pitch, radius, g_2d, g_1d);
	t.stop();
	//SAFE_CALL(cudaDeviceSynchronize(), "CUDA DEVICE SYNCHRONIZE");
	t.printTime();

	SAFE_CALL(cudaMemcpy2D(output.data, width, GPU_output, gpu_image_pitch, width, height, cudaMemcpyDeviceToHost), "CUDA MEMCPY2D DEVICE TO HOST");
	//SAFE_CALL(cudaMemcpy(output.data, GPU_output, gpu_image_pitch, cudaMemcpyDeviceToHost), "CUDA MEMCPY DEVICE TO HOST");

	//SAFE_CALL(cudaUnbindTexture(CUDA_Kernel), "CUDA UNBIND TEXTURE");
	//SAFE_CALL(cudaUnbindTexture(CUDA_Gaussian1D), "CUDA UNBIND TEXTURE");

	//SAFE_CALL(cudaUnbindTexture(CUDA_Data), "CUDA UNBIND TEXTURE");
	SAFE_CALL(cudaUnbindTexture(CUDA_Frame), "CUDA UNBIND TEXTURE");

	SAFE_CALL(cudaFree(g_2d), "CUDA FREE");
	SAFE_CALL(cudaFree(g_1d), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_input), "CUDA FREE");
	SAFE_CALL(cudaFree(GPU_output), "CUDA FREE");
}


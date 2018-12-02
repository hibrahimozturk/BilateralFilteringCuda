#ifndef BILATERAL_KERNEL_CU
#define BILATERAL_KERNEL_CU

#include <stdio.h>
#include <math.h>

#include "ProcessingInterface.h"

class CUDABilateralFilter : public ProcessingInterface {
public:
	CUDABilateralFilter(const int raio, const float sigma_s, const float sigma_r);
	~CUDABilateralFilter();
	void apply(const Mat& input, Mat& output);
protected:
	int radius;
	float ss, sr;
	//float *kernel, *gauss_1D;
};

//extern "C" void Bilateral_Caller(const unsigned char *input, unsigned char *output, int width, int height, int radius, int ss, int sr);

#endif


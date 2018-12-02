#ifndef PROCESSINGINTERFACE_H
#define PROCESSINGINTERFACE_H

#include <opencv2/core/core.hpp>

using cv::Mat;

class ProcessingInterface{
public:
	ProcessingInterface(){}
	~ProcessingInterface(){}
	virtual void apply(const Mat& input, Mat& output)=0;
};

#endif

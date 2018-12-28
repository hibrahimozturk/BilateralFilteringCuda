#include "Timer.h"

Timer::Timer(){
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
}

void Timer::start(){
	cudaEventRecord(begin);
}

void Timer::stop(){
	cudaEventRecord(end);
}

void Timer::printTime(){
	cudaEventSynchronize(end);
	float tempo=0;
	cudaEventElapsedTime(&tempo, begin, end);
	cout << tempo << endl;//" ms" << endl;
}

float Timer::getTime(){
	cudaEventSynchronize(end);
	float tempo=0;
	cudaEventElapsedTime(&tempo, begin, end);
	return float(tempo);
}


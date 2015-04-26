#ifndef CAFFE_UTIL_MATRIX_QUANTIZATION_H_
#define CAFFE_UTIL_MATRIX_QUANTIZATION_H_

#include "glog/logging.h"

#include "caffe/common.hpp"

namespace caffe {

//template<typename Dtype>
//__global__ void AssembleMatrix(const int nthreads, const Dtype* centers_data,
//		const unsigned short* indices_data, int num_center, int num_seg, int mat_height,
//		int mat_width, Dtype* mat_data);

template<typename Dtype>
__global__ void AssembleMatrix(const int nthreads, const Dtype* centers_data,
		const unsigned char* indices_data, int num_center, int num_seg, int mat_height,
		int mat_width, Dtype* mat_data);

} // namespace caffe

#endif // CAFFE_UTIL_MATRIX_QUANTIZATION_H_

//  @zyan: define commonly used device functions
#ifndef CAFFE_COMMON_MATH_CUH_
#define CAFFE_COMMON_MATH_CUH_

#include <cuda.h>

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template<typename Dtype>
static __inline__ __device__ Dtype sigmoid_dev(Dtype x);

template<>
__inline__ __device__ float sigmoid_dev(float x) {
  return 1. / (1. + expf(-x));
}

template<>
__inline__ __device__ double sigmoid_dev(double x) {
  return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype sigmoid_diff_y(Dtype y) {
  return y * (1.0 - y);
}

template<typename Dtype>
static __inline__ __device__ Dtype sigmoid_diff_y_dev(Dtype y) {
  return y * (1.0 - y);
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid<Dtype>(2. * x) - 1.;
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_dev(Dtype x) {
  return 2. * sigmoid_dev<Dtype>(2. * x) - 1.;
}

template<typename Dtype>
inline Dtype tanh_diff_x(Dtype x) {
  Dtype y = tanh<Dtype>(x);
  return 1.0 - y * y;
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_diff_x_dev(Dtype x) {
  Dtype y = tanh_dev<Dtype>(x);
  return 1.0 - y * y;
}

template<typename Dtype>
inline Dtype tanh_diff_y(Dtype y) {
  return 1.0 - y * y;
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_diff_y_dev(Dtype y) {
  return 1.0 - y * y;
}

static __inline__ __device__ int blob_offset(int channels, int height,
    int width, int n, int ch, int y, int x) {
  return ((n * channels + ch) * height + y) * width + x;
}
}  //  namespace caffe

#endif  //  #ifndef CAFFE_COMMON_MATH_CUH_

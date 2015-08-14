#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_math.cuh"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/dimension_transpose_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void forward_pass(int dim0,
    int dim1, int dim2, int dim3, const Dtype *bottom_data,
    Dtype *top_data) {
  CUDA_KERNEL_LOOP(index, dim0 * dim1 * dim2 * dim3) {
    int d0 = index / (dim1 * dim2 * dim3);
    int rm0 = index % (dim1 * dim2 * dim3);
    int d1 = rm0 / (dim2 * dim3);
    int rm1 = rm0 % (dim2 * dim3);
    int d2 = rm1 / dim3;
    int d3 = rm1 % dim3;
    int bottom_index = blob_offset(dim3, dim0, dim1, d2, d3, d0, d1);
    top_data[index] = bottom_data[bottom_index];
  }
}

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype *top_data = top[0]->mutable_gpu_data();
  const Dtype *bottom_data = bottom[0]->gpu_data();

  forward_pass<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),
      CAFFE_CUDA_NUM_THREADS>>>(top[0]->shape(0), top[0]->shape(1),
          top[0]->shape(2), top[0]->shape(3), bottom_data,
          top_data);
}

template<typename Dtype>
__global__ void backward_pass(int dim0,
    int dim1, int dim2, int dim3, Dtype *bottom_diff,
    const Dtype *top_diff) {
  CUDA_KERNEL_LOOP(index, dim0 * dim1 * dim2 * dim3) {
    int d0 = index / (dim1 * dim2 * dim3);
    int rm0 = index % (dim1 * dim2 * dim3);
    int d1 = rm0 / (dim2 * dim3);
    int rm1 = rm0 % (dim2 * dim3);
    int d2 = rm1 / dim3;
    int d3 = rm1 % dim3;
    int top_index = blob_offset(dim3, dim0, dim1, d2, d3, d0, d1);
    bottom_diff[index] = top_diff[top_index];
  }
}

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  backward_pass<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),
      CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->shape(0), bottom[0]->shape(1),
          bottom[0]->shape(2), bottom[0]->shape(3),
          bottom_diff, top_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DimensionTransposeLayer);
}  //  namespace caffe

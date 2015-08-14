#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/dimension_transpose_layer.hpp"

namespace caffe {

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  dir_ = this->layer_param_.dimension_transpose_param().direction();
}

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4);

  if (dir_
      == DimensionTransposeParameter_Direction_NUM_FIRST_TO_HEIGHT_FIRST) {
    num_ = bottom[0]->shape(0);
    channels_ = bottom[0]->shape(1);
    height_ = bottom[0]->shape(2);
    width_ = bottom[0]->shape(3);

    top_shape[0] = height_;
    top_shape[1] = width_;
    top_shape[2] = num_;
    top_shape[3] = channels_;
  } else {
    height_ = bottom[0]->shape(0);
    width_ = bottom[0]->shape(1);
    num_ = bottom[0]->shape(2);
    channels_ = bottom[0]->shape(3);

    top_shape[0] = num_;
    top_shape[1] = channels_;
    top_shape[2] = height_;
    top_shape[3] = width_;
  }
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype *bottom_data = bottom[0]->cpu_data();

  int top_index = 0;
  for (int d0 = 0; d0 < top[0]->shape(0); d0++) {
    for (int d1 = 0; d1 < top[0]->shape(1); d1++) {
      for (int d2 = 0; d2 < top[0]->shape(2); d2++) {
        for (int d3 = 0; d3 < top[0]->shape(3); d3++) {
          int bottom_index = bottom[0]->offset(d2, d3, d0, d1);
          top_data[top_index++] = bottom_data[bottom_index];
        }
      }
    }
  }
}

template<typename Dtype>
void DimensionTransposeLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();

    int bottom_index = 0;
    for (int d0 = 0; d0 < bottom[0]->shape(0); d0++) {
      for (int d1 = 0; d1 < bottom[0]->shape(1); d1++) {
        for (int d2 = 0; d2 < bottom[0]->shape(2); d2++) {
          for (int d3 = 0; d3 < bottom[0]->shape(3); d3++) {
            int top_index = top[0]->offset(d2, d3, d0, d1);
            bottom_diff[bottom_index++] = top_diff[top_index];
          }
        }
      }
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(DimensionTransposeLayer);
#endif

INSTANTIATE_CLASS(DimensionTransposeLayer);
REGISTER_LAYER_CLASS(DimensionTranspose);
}  //  namespace caffe

#ifndef CAFFE_DIMENSION_TRANSPOSE_LAYER_HPP_
#define CAFFE_DIMENSION_TRANSPOSE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/* Transpose the dimensions in the bottom blob
 * and save the result in the top blob
 * Currently only support transpose dimensions
 * either
 * 1) from (num, channels, height, width) to
 * (height, width, num, channels)
 * or
 * 2) from (height, width, num, channels) to
 * (num, channels, height, width)
 * */
template<typename Dtype>
class DimensionTransposeLayer: public Layer<Dtype> {
public:
  explicit DimensionTransposeLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const {
    return "DimensionTranspose";
  }
  /* @brief Input is a blob of shape (num, channels, height, width)
   * */
  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }
  /* @brief Output a single blob consisting of 2 stacked layers of hidden states
   * */
  virtual inline int ExactNumTopBlobs() const {
    return 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  DimensionTransposeParameter::Direction dir_;
  int num_, channels_, height_, width_;

};

}

#endif  //  #ifndef CAFFE_DIMENSION_TRANSPOSE_LAYER_HPP_

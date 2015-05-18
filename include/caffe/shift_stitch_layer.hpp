// Copyright 2015 Zhicheng Yan

#ifndef CAFFE_SHIFT_STITCH_LAYER_HPP_
#define CAFFE_SHIFT_STITCH_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ShiftStitchLayer : public Layer<Dtype> {
 public:
  explicit ShiftStitchLayer(const LayerParameter& param, int replica_id, Net<Dtype> *net)
      : Layer<Dtype>(param,replica_id,net) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ShiftStitch"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int iter_;
  std::vector<int> stride_h_;
  std::vector<int> stride_w_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int out_num_;
  int out_height_;
  int out_width_;
};

} // namespace caffe

#endif // CAFFE_SHIFT_STITCH_LAYER_HPP_

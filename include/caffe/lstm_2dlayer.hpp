#ifndef CAFFE_LSTM_2D_LAYER_HPP_
#define CAFFE_LSTM_2D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class LSTM_2DLayer: public Layer<Dtype> {
public:
  explicit LSTM_2DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTM_2D"; }
  /* @brief Input is a blob of shape (c, h, w)
   * */
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  /* @brief Output a single blob consisting of 4 stacked layers of hidden states
   * */
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int patch_h_;
  int patch_w_;
  int num_output_;

  int num_;
  int channels_;
  int patch_ny_;
  int patch_nx_;
  int num_blobs_per_dir_;

  Blob<Dtype> bias_multiplier_;
  Dtype forget_gate_scaling_factor_;

  vector<shared_ptr<Blob<Dtype> > > X_;
  vector<shared_ptr<Blob<Dtype> > > H_;
  vector<shared_ptr<Blob<Dtype> > > C_;

  vector<shared_ptr<Blob<Dtype> > > T1_;
  vector<shared_ptr<Blob<Dtype> > > T2_;
  vector<shared_ptr<Blob<Dtype> > > T3_;

  vector<shared_ptr<Blob<Dtype> > > grad1_;
  vector<shared_ptr<Blob<Dtype> > > grad2_;
  vector<shared_ptr<Blob<Dtype> > > grad3_;
  vector<shared_ptr<Blob<Dtype> > > grad4_;
  vector<shared_ptr<Blob<Dtype> > > grad5_;
  vector<shared_ptr<Blob<Dtype> > > grad6_;
};

} // namespace caffe

#endif // #ifndef CAFFE_LSTM_2D_LAYER_HPP_

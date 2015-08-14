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
#include "caffe/internal_thread.hpp"

namespace caffe {

template<typename Dtype>
class LSTM_2DLayer;

template<typename Dtype>
class LSTM_2DLayer_Forward_Worker: public InternalThread {
 public:
  LSTM_2DLayer_Forward_Worker(int dir, LSTM_2DLayer<Dtype> *layer,
      Blob<Dtype>* bottom, Blob<Dtype>* top);
  ~LSTM_2DLayer_Forward_Worker() {}

  // The thread's function
  virtual void InternalThreadEntry();
 protected:
  int dir_;
  LSTM_2DLayer<Dtype> *layer_;
  Blob<Dtype> *bottom_;
  Blob<Dtype> *top_;
};

template<typename Dtype>
class LSTM_2DLayer_Backward_Worker: public InternalThread {
 public:
  LSTM_2DLayer_Backward_Worker(int dir, LSTM_2DLayer<Dtype> *layer,
      Blob<Dtype>* bottom, Blob<Dtype>* top);
  ~LSTM_2DLayer_Backward_Worker() {}
  const Blob<Dtype>* get_bottom_diff() { return bottom_diff_.get(); }

  // The thread's function
  virtual void InternalThreadEntry();
 protected:
  int dir_;
  LSTM_2DLayer<Dtype> *layer_;
  Blob<Dtype> *bottom_;
  Blob<Dtype> *top_;
  // bottom_diff_ is used to hold gradients w.r.t. bottom for each worker
  shared_ptr< Blob<Dtype> > bottom_diff_;
};

template<typename Dtype>
class LSTM_2DLayer: public Layer<Dtype> {
public:
  explicit LSTM_2DLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTM_2D"; }
  /* @brief Input is a blob of shape (number, channels, height, width)
   * */
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  /* @brief Output a single blob consisting of
   *        4 stacked layers of hidden states
   *        Output shape is (number, 4 * num_output, out_height, out_width)
   * */
  virtual inline int ExactNumTopBlobs() const { return 1; }

  friend class LSTM_2DLayer_Forward_Worker<Dtype>;
  friend class LSTM_2DLayer_Backward_Worker<Dtype>;

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector< shared_ptr<LSTM_2DLayer_Forward_Worker<Dtype> > > forward_workers_;
  vector< shared_ptr<LSTM_2DLayer_Backward_Worker<Dtype> > > backward_workers_;

  int patch_h_;
  int patch_w_;
  bool peephole_;
  int num_output_;
  int num_;
  int channels_;
  int patch_dim_;
  int patch_ny_;
  int patch_nx_;
  int num_blobs_per_dir_;
  Blob<Dtype> bias_multiplier_;
  Dtype forget_gate_scaling_factor_;

  vector<shared_ptr< Blob<Dtype> > > X_Hx_Hy_data_, X_Hx_Hy_same_row_diff_,
  X_Hx_Hy_next_row_diff_;
  vector<shared_ptr< Blob<Dtype> > > gi_data_, gi_same_row_diff_, gi_next_row_diff_;
  vector<shared_ptr< Blob<Dtype> > > ci_data_, ci_diff_;
  vector<shared_ptr< Blob<Dtype> > > go_data_, go_diff_;
  vector<shared_ptr< Blob<Dtype> > > gfx_data_, gfx_same_row_diff_;
  vector<shared_ptr< Blob<Dtype> > > gfy_data_, gfy_same_row_diff_, gfy_next_row_diff_;
  vector<shared_ptr< Blob<Dtype> > > cstate_data_, cstate_same_row_diff_,
  cstate_next_row_diff_;
  vector<shared_ptr< Blob<Dtype> > > hidden_same_row_data_,
  hidden_prev_row_data_, hidden_diff_;
};

} // namespace caffe

#endif // #ifndef CAFFE_LSTM_2D_LAYER_HPP_

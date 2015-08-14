#ifndef CAFFE_GRID_LSTM_3D_LAYER_HPP_
#define CAFFE_GRID_LSTM_3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/* implement the Grid LSTM layer in paper
 * Grid Long Short-Term Memory"
 * URL: http://arxiv.org/abs/1507.01526
 * More specifically, implement the 3-LSTM layer
 * used in the experiment of MNIST digit
 * recognition
 * */
template<typename Dtype>
class GridLSTM3DLayer: public Layer<Dtype> {
public:
  explicit GridLSTM3DLayer(const LayerParameter& param) :
      Layer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "GridLSTM3D";
  }
  /* @brief Inputs include two blobs
   *    blob 0: used as LSTM state (memory) vector.
   *    blob 1: used as LSTM hidden output vector
   *    both blobs have shape (height, width, number, # of LSTM cells)
   * */
  virtual inline int ExactNumBottomBlobs() const {
    return 2;
  }
  /* @brief Output two blobs
   *    blob 0: LSTM state (memory) vector along depth dimension
   *    blob 1: LSTM hidden output vector along depth dimension
   *    Both blobs have shape (height, width, number, # of LSTM cells)
   * */
  virtual inline int ExactNumTopBlobs() const {
    return 2;
  }
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  GridLSTM3DParameter::Direction x_dir_;
  GridLSTM3DParameter::Direction y_dir_;
  int num_output_;
  bool peephole_;
  int img_height_;
  int img_width_;
  int num_;
  int num_blobs_per_dimension_;

  Blob<Dtype> bias_multiplier_;

  Blob<Dtype> H_data_, H_diff_;
  vector<shared_ptr<Blob<Dtype> > > gi_data_, gi_same_row_diff_, gi_next_row_diff_;
  vector<shared_ptr<Blob<Dtype> > > ci_data_, ci_diff_;
  vector<shared_ptr<Blob<Dtype> > > go_data_, go_diff_;
  vector<shared_ptr<Blob<Dtype> > > gf_data_, gf_same_row_diff_, gf_next_row_diff_;
  vector<shared_ptr<Blob<Dtype> > > cstate_data_, cstate_same_row_diff_,
  cstate_next_row_diff_;
  vector<shared_ptr<Blob<Dtype> > > hidden_same_row_data_, hidden_prev_row_data_;
  vector<shared_ptr<Blob<Dtype> > > hidden_diff_;

};

}  //  namespace caffe

#endif //  #ifndef CAFFE_GRID_LSTM_3D_LAYER_HPP_

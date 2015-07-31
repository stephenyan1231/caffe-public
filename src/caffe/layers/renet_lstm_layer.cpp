#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/renet_lstm_layer.hpp"

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype sigmoid_diff_y(Dtype y) {
  return y * (1.0 - y);
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid<Dtype>(2. * x) - 1.;
}

template<typename Dtype>
inline Dtype tanh_diff_x(Dtype x) {
  Dtype y = tanh<Dtype>(x);
  return 1.0 - y * y;
}

template<typename Dtype>
inline Dtype tanh_diff_y(Dtype y) {
  return 1.0 - y * y;
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  dir_ = this->layer_param_.renet_lstm_param().direction();
  num_output_ = this->layer_param_.renet_lstm_param().num_output();
  patch_h_ = this->layer_param_.renet_lstm_param().patch_height();
  patch_w_ = this->layer_param_.renet_lstm_param().patch_width();

  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
  CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

  channels_ = bottom[0]->shape(1);
  patch_dim_ = channels_ * patch_h_ * patch_w_;
  // two opposite scanning directions
  X_H_data_.resize(2);
  X_H_diff_.resize(2);
  gi_data_.resize(2);
  gi_diff_.resize(2);
  ci_data_.resize(2);
  ci_diff_.resize(2);
  go_data_.resize(2);
  go_diff_.resize(2);
  gf_data_.resize(2);
  gf_diff_.resize(2);
  cstate_data_.resize(2);
  cstate_diff_.resize(2);
  cstate_next_diff_.resize(2);
  hidden_data_.resize(2);
  hidden_diff_.resize(2);
  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    X_H_data_[dir_num].reset(new Blob<Dtype>());
    X_H_diff_[dir_num].reset(new Blob<Dtype>());
    gi_data_[dir_num].reset(new Blob<Dtype>());
    gi_diff_[dir_num].reset(new Blob<Dtype>());
    ci_data_[dir_num].reset(new Blob<Dtype>());
    ci_diff_[dir_num].reset(new Blob<Dtype>());
    go_data_[dir_num].reset(new Blob<Dtype>());
    go_diff_[dir_num].reset(new Blob<Dtype>());
    gf_data_[dir_num].reset(new Blob<Dtype>());
    gf_diff_[dir_num].reset(new Blob<Dtype>());
    cstate_data_[dir_num].reset(new Blob<Dtype>());
    cstate_diff_[dir_num].reset(new Blob<Dtype>());
    cstate_next_diff_[dir_num].reset(new Blob<Dtype>());
    hidden_data_[dir_num].reset(new Blob<Dtype>());
    hidden_diff_[dir_num].reset(new Blob<Dtype>());
  }

  // four paramater matrices W_i, W_c, W_o, W_f
  // four bias vectors b_i, b_c, b_o, b_f
  num_blobs_per_dir_ = 8;
  this->blobs_.resize(2 * num_blobs_per_dir_);
  // four paramater matrices
  // W_i = [W_{i,x}, H_i]
  // W_c = [W_{c,x}, H_c]
  // W_o = [W_{o,x}, H_o]
  // W_f = [W_{f,x}, H_f]
  vector<int> W_shape(2);
  W_shape[0] = num_output_;
  W_shape[1] = patch_dim_ + num_output_;
  // four bias vectors b_i, b_c, b_o, b_f
  vector<int> B_shape(1, num_output_);

  shared_ptr<Filler<Dtype> > general_weight_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().general_weight_filler()));
  shared_ptr<Filler<Dtype> > general_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().general_bias_filler()));
  shared_ptr<Filler<Dtype> > forget_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().forget_gate_bias_filler()));

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    // four paramater matrices, W_i, W_c, W_o, W_f
    for (int p = 0; p < 4; ++p) {
      this->blobs_[dir_num * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(W_shape));
    }
    // four bias vectors, b_i, b_c, b_o, b_f
    for (int p = 4; p < 8; ++p) {
      this->blobs_[dir_num * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(B_shape));
    }
    // four paramater matrices, W_i, W_c, W_o, W_f
    for (int p = 0; p < 4; ++p) {
      general_weight_filler->Fill(
          this->blobs_[dir_num * num_blobs_per_dir_ + p].get());
    }
    // three bias vectors, b_i, b_c, b_o
    for (int p = 4; p < 7; ++p) {
      general_bias_filler->Fill(
          this->blobs_[dir_num * num_blobs_per_dir_ + p].get());
    }
    // forget gate bias vector, b_f
    for (int p = 7; p < 8; ++p) {
      forget_gate_bias_filler->Fill(
          this->blobs_[dir_num * num_blobs_per_dir_ + p].get());
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[0]->shape(1), channels_);
  CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
  CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

  num_ = bottom[0]->shape(0);
  patch_ny_ = bottom[0]->shape(2) / patch_h_;
  patch_nx_ = bottom[0]->shape(3) / patch_w_;

  num_RNN_ =
      dir_ == ReNetLSTMParameter_Direction_X_DIR ? patch_ny_ : patch_nx_;
  num_steps_ =
      dir_ == ReNetLSTMParameter_Direction_X_DIR ? patch_nx_ : patch_ny_;

  vector<int> X_H_shape_4D(4);
  X_H_shape_4D[0] = num_steps_;
  X_H_shape_4D[1] = num_RNN_;
  X_H_shape_4D[2] = num_;
  X_H_shape_4D[3] = patch_dim_ + num_output_;

  vector<int> X_H_shape_3D(3);
  X_H_shape_3D[0] = num_RNN_;
  X_H_shape_3D[1] = num_;
  X_H_shape_3D[2] = patch_dim_ + num_output_;

  vector<int> cell_shape_4D(4);
  cell_shape_4D[0] = num_steps_;
  cell_shape_4D[1] = num_RNN_;
  cell_shape_4D[2] = num_;
  cell_shape_4D[3] = num_output_;

  vector<int> cell_shape_3D(3);
  cell_shape_3D[0] = num_RNN_;
  cell_shape_3D[1] = num_;
  cell_shape_3D[2] = num_output_;

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    X_H_data_[dir_num]->Reshape(X_H_shape_4D);
    X_H_diff_[dir_num]->Reshape(X_H_shape_3D);
    gi_data_[dir_num]->Reshape(cell_shape_4D);
    gi_diff_[dir_num]->Reshape(cell_shape_3D);
    ci_data_[dir_num]->Reshape(cell_shape_4D);
    ci_diff_[dir_num]->Reshape(cell_shape_3D);
    go_data_[dir_num]->Reshape(cell_shape_4D);
    go_diff_[dir_num]->Reshape(cell_shape_3D);
    gf_data_[dir_num]->Reshape(cell_shape_4D);
    gf_diff_[dir_num]->Reshape(cell_shape_3D);
    cstate_data_[dir_num]->Reshape(cell_shape_4D);
    cstate_diff_[dir_num]->Reshape(cell_shape_3D);
    cstate_next_diff_[dir_num]->Reshape(cell_shape_3D);
    hidden_data_[dir_num]->Reshape(cell_shape_3D);
    hidden_diff_[dir_num]->Reshape(cell_shape_3D);
  }

  vector<int> top_shape(4);
  top_shape[0] = num_;
  top_shape[1] = 2 * num_output_;
  top_shape[2] = patch_ny_;
  top_shape[3] = patch_nx_;
  top[0]->Reshape(top_shape);

  vector<int> bias_shape(1, num_RNN_ * num_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set<Dtype>(num_RNN_ * num_, Dtype(1),
      bias_multiplier_.mutable_cpu_data());
}

// fill in X_H data
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Fill_X_H_Data_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* bottom) {
  bool not_start = step_id != step_start;
  const Dtype* bottom_data = bottom->cpu_data();
  Dtype *X_H_data = X_H_data_[dir_num]->mutable_cpu_data()
      + X_H_data_[dir_num]->offset(step_id);

  const Dtype *hidden_data = hidden_data_[dir_num]->cpu_data();

  int X_H_data_index = 0;
  int hidden_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
        for (int py = 0; py < patch_h_; ++py) {
          for (int px = 0; px < patch_w_; ++px) {
            int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
            int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
            int bottom_index = bottom->offset(n, ch, y * patch_h_ + py,
                x * patch_w_ + px);
            X_H_data[X_H_data_index++] = bottom_data[bottom_index];
          }
        }
      }
      // fill X_H with previous hidden outputs
      for (int d = 0; d < num_output_; ++d) {
        if (!not_start) {
          X_H_data[X_H_data_index++] = 0;
        } else {
          X_H_data[X_H_data_index++] = hidden_data[hidden_index++];
        }
      }
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellData_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* top) {
  int step = dir_num == 0 ? 1 : -1;
  Dtype* top_data = top->mutable_cpu_data();
  const Dtype* X_H_data = X_H_data_[dir_num]->cpu_data()
      + X_H_data_[dir_num]->offset(step_id);

  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->cpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->cpu_data();

  Dtype* gi_data = gi_data_[dir_num]->mutable_cpu_data()
      + gi_data_[dir_num]->offset(step_id);
  Dtype* ci_data = ci_data_[dir_num]->mutable_cpu_data()
      + ci_data_[dir_num]->offset(step_id);
  Dtype* go_data = go_data_[dir_num]->mutable_cpu_data()
      + go_data_[dir_num]->offset(step_id);
  Dtype* gf_data = gf_data_[dir_num]->mutable_cpu_data()
      + gf_data_[dir_num]->offset(step_id);
  Dtype* cstate_data = cstate_data_[dir_num]->mutable_cpu_data()
      + cstate_data_[dir_num]->offset(step_id);
  Dtype* hidden_data = hidden_data_[dir_num]->mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_i_data,
      (Dtype) 0., gi_data);
  // add bias
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 4]->cpu_data(), (Dtype) 1.,
      gi_data);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_c_data,
      (Dtype) 0., ci_data);
  // add bias
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 5]->cpu_data(), (Dtype) 1.,
      ci_data);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_o_data,
      (Dtype) 0., go_data);
  // add bias
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 6]->cpu_data(), (Dtype) 1.,
      go_data);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_f_data,
      (Dtype) 0., gf_data);
  // add bias
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->cpu_data(), (Dtype) 1.,
      gf_data);

  bool not_start = step_id != step_start;

  const Dtype* cstate_data_prev_ptr = NULL;
  if (not_start) {
    cstate_data_prev_ptr = cstate_data_[dir_num]->cpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }

  int data_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        gi_data[data_index] = sigmoid<Dtype>(gi_data[data_index]);
        ci_data[data_index] = tanh<Dtype>(ci_data[data_index]);
        go_data[data_index] = sigmoid<Dtype>(go_data[data_index]);
        gf_data[data_index] = sigmoid<Dtype>(gf_data[data_index]);
        cstate_data[data_index] = ci_data[data_index] * gi_data[data_index];
        if (not_start) {
          cstate_data[data_index] += gf_data[data_index]
              * cstate_data_prev_ptr[data_index];
        }
        hidden_data[data_index] = go_data[data_index]
            * tanh<Dtype>(cstate_data[data_index]);
        // copy hidden output into top data
        int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
        int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
        top_data[top->offset(n, dir_num * num_output_ + d, y, x)] =
            hidden_data[data_index];
        data_index++;
      }  // for (int d = 0; d < num_output_; ++d)
    }  // for (int n = 0; n < num_; ++n)
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    int step_start, step_end, step_min, step_max, step;
    if (dir_ == ReNetLSTMParameter_Direction_X_DIR) {
      step_start = dir_num == 0 ? 0 : patch_nx_ - 1;
      step_end = dir_num == 0 ? patch_nx_ - 1 : 0;
    } else {
      step_start = dir_num == 0 ? 0 : patch_ny_ - 1;
      step_end = dir_num == 0 ? patch_ny_ - 1 : 0;
    }
    step_min = step_start <= step_end ? step_start : step_end;
    step_max = step_start <= step_end ? step_end : step_start;
    step = dir_num == 0 ? 1 : -1;
    for (int step_id = step_start; step_id >= step_min && step_id <= step_max;
        step_id += step) {
      Fill_X_H_Data_cpu(dir_num, step_id, step_start, bottom[0]);
      ComputeCellData_cpu(dir_num, step_id, step_start, top[0]);
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::FillHiddenDiff_cpu(int dir_num, int step_id,
    int step_end, Blob<Dtype>* top) {
  const Dtype* top_diff = top->cpu_diff();
  Dtype* hidden_diff = hidden_diff_[dir_num]->mutable_cpu_data();
  int data_index = 0;
  bool not_end = step_id != step_end;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      const Dtype* X_H_diff = X_H_diff_[dir_num]->cpu_data()
          + X_H_diff_[dir_num]->offset(RNN, n);
      for (int d = 0; d < num_output_; ++d) {
        // copy top diff into hidden_diff
        int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
        int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
        hidden_diff[data_index] = top_diff[top->offset(n,
            dir_num * num_output_ + d, y, x)];
        if (not_end) {
          hidden_diff[data_index] += X_H_diff[patch_dim_ + d];
        }
        data_index++;
      }
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellDiff_cpu(int dir_num, int step_id,
    int step_start, int step_end) {
  const int step = dir_num == 0 ? 1 : -1;
  const int cell_data_offset = gi_data_[dir_num]->offset(step_id);

  const Dtype* hidden_diff = hidden_diff_[dir_num]->mutable_cpu_data();

  const Dtype* gi_data = gi_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* gi_diff = gi_diff_[dir_num]->mutable_cpu_data();

  const Dtype* ci_data = ci_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* ci_diff = ci_diff_[dir_num]->mutable_cpu_data();

  const Dtype* go_data = go_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* go_diff = go_diff_[dir_num]->mutable_cpu_data();

  const Dtype* gf_data = gf_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* gf_diff = gf_diff_[dir_num]->mutable_cpu_data();

  const Dtype* cstate_data = cstate_data_[dir_num]->cpu_data()
      + cell_data_offset;
  Dtype* cstate_diff = cstate_diff_[dir_num]->mutable_cpu_data();
  Dtype* cstate_next_diff = cstate_next_diff_[dir_num]->mutable_cpu_data();

  bool not_start = step_id != step_start;
  bool not_end = step_id != step_end;

  const Dtype* gf_next_data_ptr = NULL;
  if (not_end) {
    gf_next_data_ptr = gf_data_[dir_num]->cpu_data()
        + gf_data_[dir_num]->offset(step_id + step);
  }

  const Dtype* cstate_prev_data_ptr = NULL;
  if (not_start) {
    cstate_prev_data_ptr = cstate_data_[dir_num]->cpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }

  int data_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        Dtype cstate_val = cstate_data[data_index];
        Dtype go_val = go_data[data_index];

        go_diff[data_index] = hidden_diff[data_index] * tanh<Dtype>(cstate_val)
            * sigmoid_diff_y<Dtype>(go_val);
        cstate_diff[data_index] = hidden_diff[data_index] * go_val
            * tanh_diff_x<Dtype>(cstate_val);
        if (not_end) {
          cstate_diff[data_index] += cstate_next_diff[data_index]
              * gf_next_data_ptr[data_index];
        }
        if (not_start) {
          gf_diff[data_index] = cstate_diff[data_index]
              * cstate_prev_data_ptr[data_index]
              * sigmoid_diff_y<Dtype>(gf_data[data_index]);
        } else {
          gf_diff[data_index] = 0;
        }
        Dtype gi_val = gi_data[data_index];
        Dtype ci_val = ci_data[data_index];
        gi_diff[data_index] = cstate_diff[data_index] * ci_val
            * sigmoid_diff_y<Dtype>(gi_val);
        ci_diff[data_index] = cstate_diff[data_index] * gi_val
            * tanh_diff_y<Dtype>(ci_val);
        cstate_next_diff[data_index] = cstate_diff[data_index];
        data_index++;
      }
    }
  }
}

// compute gradient w.r.t X_H
// compute gradient w.r.t. bottom
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Compute_X_H_Diff_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* bottom) {
  int X_H_dim = patch_dim_ + num_output_;
  const Dtype* gi_diff = gi_diff_[dir_num]->cpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->cpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->cpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->cpu_data();

  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->cpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->cpu_data();

  // compute gradients w.r.t. X_H_
  Dtype* X_H_diff = X_H_diff_[dir_num]->mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., gi_diff, param_W_i_data, (Dtype) 0., X_H_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., ci_diff, param_W_c_data, (Dtype) 1., X_H_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., go_diff, param_W_o_data, (Dtype) 1., X_H_diff);
  if (step_id != step_start) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
        num_output_, (Dtype) 1., gf_diff, param_W_f_data, (Dtype) 1.,
        X_H_diff);
  }

  // copy gradients w.r.t. X_H_ into bottom diff
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      X_H_diff = X_H_diff_[dir_num]->mutable_cpu_data()
          + X_H_diff_[dir_num]->offset(RNN, n);
      int data_index = 0;
      for (int ch = 0; ch < channels_; ++ch) {
        for (int py = 0; py < patch_h_; ++py) {
          for (int px = 0; px < patch_w_; ++px) {
            int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
            int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
            bottom_diff[bottom->offset(n, ch, y * patch_h_ + py,
                x * patch_w_ + px)] += X_H_diff[data_index++];
          }
        }
      }
    }
  }
}

// compute gradients w.r.t. parameters and biases
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeParamDiff_cpu(int dir_num, int step_id,
    int step_start) {
  const int X_H_dim = patch_dim_ + num_output_;
  const Dtype* X_H_data = X_H_data_[dir_num]->cpu_data()
      + X_H_data_[dir_num]->offset(step_id);

  const Dtype* gi_diff = gi_diff_[dir_num]->cpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->cpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->cpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->cpu_data();

  Dtype* param_W_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_]->mutable_cpu_diff();
  Dtype* param_W_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->mutable_cpu_diff();
  Dtype* param_W_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->mutable_cpu_diff();
  Dtype* param_W_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->mutable_cpu_diff();

  Dtype* bias_b_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 4]->mutable_cpu_diff();
  Dtype* bias_b_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 5]->mutable_cpu_diff();
  Dtype* bias_b_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 6]->mutable_cpu_diff();
  Dtype* bias_b_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->mutable_cpu_diff();

  bool not_start = step_id != step_start;

  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    // compute gradients w.r.t. parameters
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., gi_diff, X_H_data, (Dtype) 1., param_W_i_diff);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., ci_diff, X_H_data, (Dtype) 1., param_W_c_diff);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., go_diff, X_H_data, (Dtype) 1., param_W_o_diff);
    if (not_start) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim,
          num_, (Dtype) 1., gf_diff, X_H_data, (Dtype) 1., param_W_f_diff);
    }

    // compute gradients w.r.t. biases
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gi_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_i_diff);
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., ci_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_c_diff);
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., go_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_o_diff);
    if (not_start) {
      caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gf_diff,
          bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_f_diff);
    }
    X_H_data += num_ * X_H_dim;
    gi_diff += num_ * num_output_;
    ci_diff += num_ * num_output_;
    go_diff += num_ * num_output_;
    gf_diff += num_ * num_output_;
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(), 0, bottom_diff);

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    for (int i = 0; i < num_blobs_per_dir_; ++i) {
      Dtype* param_diff =
          this->blobs_[dir_num * num_blobs_per_dir_ + i]->mutable_cpu_diff();
      caffe_set<Dtype>(this->blobs_[dir_num * num_blobs_per_dir_ + i]->count(),
          0, param_diff);
    }
    int step_start, step_end, step_min, step_max, step;
    if (dir_ == ReNetLSTMParameter_Direction_X_DIR) {
      step_start = dir_num == 0 ? 0 : patch_nx_ - 1;
      step_end = dir_num == 0 ? patch_nx_ - 1 : 0;
    } else {
      step_start = dir_num == 0 ? 0 : patch_ny_ - 1;
      step_end = dir_num == 0 ? patch_ny_ - 1 : 0;
    }
    step_min = step_start <= step_end ? step_start : step_end;
    step_max = step_start <= step_end ? step_end : step_start;
    step = dir_num == 0 ? 1 : -1;
    for (int step_id = step_end; step_id >= step_min && step_id <= step_max;
        step_id -= step) {
      FillHiddenDiff_cpu(dir_num, step_id, step_end, top[0]);
      ComputeCellDiff_cpu(dir_num, step_id, step_start, step_end);
      Compute_X_H_Diff_cpu(dir_num, step_id, step_start, bottom[0]);
      ComputeParamDiff_cpu(dir_num, step_id, step_start);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReNetLSTMLayer);
#endif

INSTANTIATE_CLASS(ReNetLSTMLayer);
REGISTER_LAYER_CLASS(ReNetLSTM);
}  // namespace caffe

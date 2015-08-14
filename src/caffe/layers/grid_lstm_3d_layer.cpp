#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_math.cuh"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/grid_lstm_3D_layer.hpp"

namespace caffe {

template<typename Dtype>
void GridLSTM3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  x_dir_ = this->layer_param_.grid_lstm_3d_param().x_direction();
  y_dir_ = this->layer_param_.grid_lstm_3d_param().y_direction();
  num_output_ = this->layer_param_.grid_lstm_3d_param().num_output();
  peephole_ = this->layer_param_.grid_lstm_3d_param().peephole();

  num_blobs_per_dimension_ = 11;
  // for each dimension 'd' \in {x,y,z} out of 3 dimensions
  //  4 parameter matrices W^d_i, W^d_c, W^d_o, W^d_f
  //  3 parameter matrices W^d_{i,c}, W^d_{o,c}, W^d_{f,c}
  //  4 bias vectors b^d_i, b^d_c, b^d_o, b^d_f
  this->blobs_.resize(3 * num_blobs_per_dimension_);
  //  4 parameter matrices W^d_i, W^d_c, W^d_o, W^d_f
  vector<int> W_H_shape(2);
  W_H_shape[0] = num_output_;
  W_H_shape[1] = 3 * num_output_;
  //  3 parameter matrices W^d_{i,c}, W^d_{o,c}, W^d_{f,c}
  vector<int> W_C_shape(2);
  W_C_shape[0] = num_output_;
  W_C_shape[1] = num_output_;
  //  4 bias vectors b^d_i, b^d_c, b^d_o, b^d_f
  vector<int> B_shape(1, num_output_);

  shared_ptr<Filler<Dtype> > general_weight_filler(
      GetFiller<Dtype>(
          this->layer_param_.grid_lstm_3d_param().general_weight_filler()));
  shared_ptr<Filler<Dtype> > cell_input_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.grid_lstm_3d_param().cell_input_bias_filler()));
  shared_ptr<Filler<Dtype> > forget_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.grid_lstm_3d_param().forget_gate_bias_filler()));
  shared_ptr<Filler<Dtype> > input_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.grid_lstm_3d_param().input_gate_bias_filler()));
  shared_ptr<Filler<Dtype> > output_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.grid_lstm_3d_param().output_gate_bias_filler()));

  for (int dim = 0; dim < 3; ++dim) {
    int blob_offset = dim * num_blobs_per_dimension_;
    for (int p = 0; p < 4; ++p) {
      this->blobs_[blob_offset + p].reset(new Blob<Dtype>(W_H_shape));
    }
    for (int p = 4; p < 7; ++p) {
      this->blobs_[blob_offset + p].reset(new Blob<Dtype>(W_C_shape));
    }
    for (int p = 7; p < 11; ++p) {
      this->blobs_[blob_offset + p].reset(new Blob<Dtype>(B_shape));
    }
    if (peephole_) {
      for (int p = 0; p < 7; ++p) {
        general_weight_filler->Fill(this->blobs_[blob_offset + p].get());
      }
    } else {
      for (int p = 0; p < 4; ++p) {
        general_weight_filler->Fill(this->blobs_[blob_offset + p].get());
      }
    }
    // input gate bias vector, b_i
    input_gate_bias_filler->Fill(this->blobs_[blob_offset + 7].get());
    // cell input bias vector, b_c
    cell_input_bias_filler->Fill(this->blobs_[blob_offset + 8].get());
    // output gate bias vector, b_0
    output_gate_bias_filler->Fill(this->blobs_[blob_offset + 9].get());
    // forget gate bias vector, b_f
    forget_gate_bias_filler->Fill(this->blobs_[blob_offset + 10].get());
  }

  gi_data_.resize(3);
  gi_same_row_diff_.resize(3);
  gi_next_row_diff_.resize(3);
  ci_data_.resize(3);
  ci_diff_.resize(3);
  go_data_.resize(3);
  go_diff_.resize(3);
  gf_data_.resize(3);
  gf_same_row_diff_.resize(3);
  gf_next_row_diff_.resize(3);
  cstate_data_.resize(3);
  cstate_same_row_diff_.resize(3);
  cstate_next_row_diff_.resize(3);
  hidden_same_row_data_.resize(3);
  hidden_prev_row_data_.resize(3);
  hidden_diff_.resize(3);
  for (int dim = 0; dim < 3; ++dim) {
    gi_data_[dim].reset(new Blob<Dtype>);
    gi_same_row_diff_[dim].reset(new Blob<Dtype>);
    gi_next_row_diff_[dim].reset(new Blob<Dtype>);
    ci_data_[dim].reset(new Blob<Dtype>);
    ci_diff_[dim].reset(new Blob<Dtype>);
    go_data_[dim].reset(new Blob<Dtype>);
    go_diff_[dim].reset(new Blob<Dtype>);
    gf_data_[dim].reset(new Blob<Dtype>);
    gf_same_row_diff_[dim].reset(new Blob<Dtype>);
    gf_next_row_diff_[dim].reset(new Blob<Dtype>);
    cstate_data_[dim].reset(new Blob<Dtype>);
    cstate_same_row_diff_[dim].reset(new Blob<Dtype>);
    cstate_next_row_diff_[dim].reset(new Blob<Dtype>);
    hidden_same_row_data_[dim].reset(new Blob<Dtype>);
    hidden_prev_row_data_[dim].reset(new Blob<Dtype>);
    hidden_diff_[dim].reset(new Blob<Dtype>);
  }
}

template<typename Dtype>
void GridLSTM3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
  }
  CHECK_EQ(num_output_, bottom[0]->shape(3));

  img_height_ = bottom[0]->shape(0);
  img_width_ = bottom[0]->shape(1);
  num_ = bottom[0]->shape(2);

  vector<int> H_shape(4);
  H_shape[0] = img_height_;
  H_shape[1] = img_width_;
  H_shape[2] = num_;
  H_shape[3] = 3 * num_output_;

  H_data_.Reshape(H_shape);
  H_diff_.Reshape(H_shape);

  vector<int> cell_shape_4D(4);
  cell_shape_4D[0] = img_height_;
  cell_shape_4D[1] = img_width_;
  cell_shape_4D[2] = num_;
  cell_shape_4D[3] = num_output_;

  vector<int> cell_shape_3D(3);
  cell_shape_3D[0] = img_width_;
  cell_shape_3D[1] = num_;
  cell_shape_3D[2] = num_output_;

  vector<int> cell_shape_2D(2);
  cell_shape_2D[0] = num_;
  cell_shape_2D[1] = num_output_;

  top[0]->Reshape(cell_shape_4D);
  top[1]->Reshape(cell_shape_4D);

  for (int dim = 0; dim < 3; ++dim) {
    gi_data_[dim]->Reshape(cell_shape_4D);
    gi_same_row_diff_[dim]->Reshape(cell_shape_3D);
    gi_next_row_diff_[dim]->Reshape(cell_shape_3D);
    ci_data_[dim]->Reshape(cell_shape_4D);
    ci_diff_[dim]->Reshape(cell_shape_2D);
    go_data_[dim]->Reshape(cell_shape_4D);
    go_diff_[dim]->Reshape(cell_shape_2D);
    gf_data_[dim]->Reshape(cell_shape_4D);
    gf_same_row_diff_[dim]->Reshape(cell_shape_3D);
    gf_next_row_diff_[dim]->Reshape(cell_shape_3D);
    cstate_data_[dim]->Reshape(cell_shape_4D);
    cstate_same_row_diff_[dim]->Reshape(cell_shape_3D);
    cstate_next_row_diff_[dim]->Reshape(cell_shape_3D);
    hidden_same_row_data_[dim]->Reshape(cell_shape_3D);
    hidden_prev_row_data_[dim]->Reshape(cell_shape_3D);
    hidden_diff_[dim]->Reshape(cell_shape_2D);
    if (dim == 2) {
      // directly write cell state data into top[0] data
      cstate_data_[dim]->set_cpu_data(top[0]->mutable_cpu_data());
    }
  }

  vector<int> bias_multiplier_shape(1, num_);
  bias_multiplier_.Reshape(bias_multiplier_shape);
  caffe_set<Dtype>(num_, Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template<typename Dtype>
void GridLSTM3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int start_y =
      y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 0 : img_height_ - 1;
  int end_y =
      y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? img_height_ - 1 : 0;
  int min_y = start_y <= end_y ? start_y : end_y;
  int max_y = start_y <= end_y ? end_y : start_y;
  int step_y = y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 1 : -1;

  int start_x =
      x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 0 : img_width_ - 1;
  int end_x =
      x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? img_width_ - 1 : 0;
  int min_x = start_x <= end_x ? start_x : end_x;
  int max_x = start_x <= end_x ? end_x : start_x;
  int step_x = x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 1 : -1;

  Dtype *H_data = H_data_.mutable_cpu_data();

  for (int y = start_y; y >= min_y && y <= max_y; y += step_y) {
    // directly write hidden output in the depth dimension into top[1] data
    hidden_same_row_data_[2]->set_cpu_data(
        top[1]->mutable_cpu_data() + top[1]->offset(y));
    if (y != start_y) {
      hidden_prev_row_data_[2]->set_cpu_data(
          top[1]->mutable_cpu_data() + top[1]->offset(y - step_y));
    }

    for (int x = start_x; x >= min_x && x <= max_x; x += step_x) {
      vector<bool> not_start(3);
      not_start[0] = x != start_x;
      not_start[1] = y != start_y;
      not_start[2] = true;

      // fill H with previous hidden outputs
      // H = [H_{z,y,x-1} H_{z,y-1,x} H_{z-1,y,x}]
      for (int n = 0; n < num_; ++n) {
        Dtype *H_data_ptr = H_data + H_data_.offset(y, x, n);
        for (int dim = 0; dim < 3; ++dim) {
          const Dtype *H_prev_data = NULL;
          if (not_start[dim]) {
            switch (dim) {
            case 0: {
              H_prev_data = hidden_same_row_data_[dim]->cpu_data()
                  + hidden_same_row_data_[dim]->offset(x - step_x, n);
              break;
            }
            case 1: {
              H_prev_data = hidden_prev_row_data_[dim]->cpu_data()
                  + hidden_prev_row_data_[dim]->offset(x, n);
              break;
            }
            case 2: {
              H_prev_data = bottom[1]->cpu_data() + bottom[1]->offset(y, x, n);
              break;
            }
            }
          }
          int offset = dim * num_output_;
          for (int d = 0; d < num_output_; ++d) {
            if (not_start[dim]) {
              H_data_ptr[offset + d] = H_prev_data[d];
            } else {
              H_data_ptr[offset + d] = 0;
            }
          }
        }
      }

      const Dtype *H_data_ptr = H_data + H_data_.offset(y, x);

      for (int dim = 0; dim < 3; ++dim) {
        int blob_offset = dim * num_blobs_per_dimension_;
        const Dtype *param_W_i_data = this->blobs_[blob_offset]->cpu_data();
        const Dtype *param_W_c_data =
            this->blobs_[blob_offset + 1]->cpu_data();
        const Dtype *param_W_o_data =
            this->blobs_[blob_offset + 2]->cpu_data();
        const Dtype *param_W_f_data =
            this->blobs_[blob_offset + 3]->cpu_data();
        const Dtype *param_W_i_c_data =
            this->blobs_[blob_offset + 4]->cpu_data();
        const Dtype *param_W_o_c_data =
            this->blobs_[blob_offset + 5]->cpu_data();
        const Dtype *param_W_f_c_data =
            this->blobs_[blob_offset + 6]->cpu_data();
        const Dtype *bias_b_i_data = this->blobs_[blob_offset + 7]->cpu_data();
        const Dtype *bias_b_c_data = this->blobs_[blob_offset + 8]->cpu_data();
        const Dtype *bias_b_o_data = this->blobs_[blob_offset + 9]->cpu_data();
        const Dtype *bias_b_f_data =
            this->blobs_[blob_offset + 10]->cpu_data();

        Dtype *gi_data = gi_data_[dim]->mutable_cpu_data()
            + gi_data_[dim]->offset(y, x);
        Dtype *ci_data = ci_data_[dim]->mutable_cpu_data()
            + ci_data_[dim]->offset(y, x);
        Dtype *go_data = go_data_[dim]->mutable_cpu_data()
            + go_data_[dim]->offset(y, x);
        Dtype *gf_data = gf_data_[dim]->mutable_cpu_data()
            + gf_data_[dim]->offset(y, x);
        Dtype *cstate_data = cstate_data_[dim]->mutable_cpu_data()
            + cstate_data_[dim]->offset(y, x);
        Dtype *hidden_data = hidden_same_row_data_[dim]->mutable_cpu_data()
            + hidden_same_row_data_[dim]->offset(x);

        const Dtype *cstate_prev_data = NULL;
        if (not_start[dim]) {
          switch (dim) {
          case 0: {
            cstate_prev_data = cstate_data_[dim]->cpu_data()
                + cstate_data_[dim]->offset(y, x - step_x);
            break;
          }
          case 1: {
            cstate_prev_data = cstate_data_[dim]->cpu_data()
                + cstate_data_[dim]->offset(y - step_y, x);
            break;
          }
          case 2: {
            cstate_prev_data = bottom[0]->cpu_data() + bottom[0]->offset(y, x);
          }
          }
        }

        // start to compute LSTM cell data

        // compute gi_data
        // W^x_i * H_{z,y,x}
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
            3 * num_output_, (Dtype) 1., H_data_ptr, param_W_i_data,
            (Dtype) 0., gi_data);
        if (not_start[dim] && peephole_) {
          // W^x_{i,c} * s_{z,y,x-1}
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
              num_output_, (Dtype) 1., cstate_prev_data, param_W_i_c_data,
              (Dtype) 1., gi_data);
        }
        // add bias b^x_i
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_, 1,
            (Dtype) 1., bias_multiplier_.cpu_data(), bias_b_i_data, (Dtype) 1.,
            gi_data);

        // compute ci_data
        // W_^x_c*H_{z,y,x}
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
            3 * num_output_, (Dtype) 1., H_data_ptr, param_W_c_data,
            (Dtype) 0., ci_data);
        // add bias b^x_c
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_, 1,
            (Dtype) 1., bias_multiplier_.cpu_data(), bias_b_c_data, (Dtype) 1.,
            ci_data);

        // compute go_data
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
            3 * num_output_, (Dtype) 1., H_data_ptr, param_W_o_data,
            (Dtype) 0., go_data);
        // add bias b^x_o
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_, 1,
            (Dtype) 1., bias_multiplier_.cpu_data(), bias_b_o_data, (Dtype) 1.,
            go_data);

        // compute gf_data
        // W^x_f*H_{z,y,x}
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
            3 * num_output_, (Dtype) 1., H_data_ptr, param_W_f_data,
            (Dtype) 0., gf_data);
        if (not_start[dim] && peephole_) {
          // W^x_{f,c} * s_{z,y,x-1}
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
              num_output_, (Dtype) 1., cstate_prev_data, param_W_f_c_data,
              (Dtype) 1., gf_data);
        }
        // add bias b^x_f
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_, 1,
            (Dtype) 1., bias_multiplier_.cpu_data(), bias_b_f_data, (Dtype) 1.,
            gf_data);

        int data_index = 0;
        for (int n = 0; n < num_; ++n) {
          for (int d = 0; d < num_output_; ++d) {
            gi_data[data_index] = sigmoid<Dtype>(gi_data[data_index]);
            ci_data[data_index] = tanh<Dtype>(ci_data[data_index]);
            gf_data[data_index] = sigmoid<Dtype>(gf_data[data_index]);
            cstate_data[data_index] = ci_data[data_index]
                * gi_data[data_index];
            if (not_start[dim]) {
              cstate_data[data_index] += gf_data[data_index]
                  * cstate_prev_data[data_index];
            }
            data_index++;
          }
        }
        if (peephole_) {
          // compute go_data
          // W^x_{o,c} * s_{z,y,x}
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_output_,
              num_output_, (Dtype) 1., cstate_data, param_W_o_c_data,
              (Dtype) 1., go_data);
        }
        data_index = 0;
        for (int n = 0; n < num_; ++n) {
          for (int d = 0; d < num_output_; ++d) {
            go_data[data_index] = sigmoid<Dtype>(go_data[data_index]);
            hidden_data[data_index] = go_data[data_index]
                * tanh<Dtype>(cstate_data[data_index]);
            data_index++;
          }  // for (int d = 0; d < num_output_; ++d)
        }  // for (int n = 0; n < num_; ++n)
      }  // for (int dim = 0; dim < 3; ++dim)
    }  // for (int x = start_x; x >= min_x && x <= max_x; x += step_x)
    caffe_copy<Dtype>(hidden_same_row_data_[0]->count(),
        hidden_same_row_data_[0]->cpu_data(),
        hidden_prev_row_data_[0]->mutable_cpu_data());
    caffe_copy<Dtype>(hidden_same_row_data_[1]->count(),
        hidden_same_row_data_[1]->cpu_data(),
        hidden_prev_row_data_[1]->mutable_cpu_data());
  }  // for (int y = start_y; y >= min_y && y <= max_y; y += step_y)
}

template<typename Dtype>
void GridLSTM3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
    caffe_set<Dtype>(bottom[i]->count(), 0, bottom_diff);
  }
  for (int dim = 0; dim < 3; ++dim) {
    int blob_offset = dim * num_blobs_per_dimension_;
    for (int i = 0; i < num_blobs_per_dimension_; ++i) {
      Dtype *param_diff = this->blobs_[blob_offset + i]->mutable_cpu_diff();
      caffe_set<Dtype>(this->blobs_[blob_offset + i]->count(), 0, param_diff);
    }
  }

  int start_y =
      y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 0 : img_height_ - 1;
  int end_y =
      y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? img_height_ - 1 : 0;
  int min_y = start_y <= end_y ? start_y : end_y;
  int max_y = start_y <= end_y ? end_y : start_y;
  int step_y = y_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 1 : -1;

  int start_x =
      x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 0 : img_width_ - 1;
  int end_x =
      x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? img_width_ - 1 : 0;
  int min_x = start_x <= end_x ? start_x : end_x;
  int max_x = start_x <= end_x ? end_x : start_x;
  int step_x = x_dir_ == GridLSTM3DParameter_Direction_POSITIVE ? 1 : -1;

  Dtype *H_diff = H_diff_.mutable_cpu_data();
  caffe_set<Dtype>(H_diff_.count(), 0, H_diff);

  const Dtype *H_data = H_data_.cpu_data();

  for (int y = end_y; y >= min_y && y <= max_y; y -= step_y) {
    caffe_set<Dtype>(cstate_same_row_diff_[0]->count(), 0,
        cstate_same_row_diff_[0]->mutable_cpu_data());
    caffe_set<Dtype>(cstate_same_row_diff_[1]->count(), 0,
        cstate_same_row_diff_[1]->mutable_cpu_data());
    // copy top cell state blob diff into cstate_diff_
    caffe_copy<Dtype>(top[0]->count() / img_height_,
        top[0]->cpu_diff() + top[0]->offset(y),
        cstate_same_row_diff_[2]->mutable_cpu_data());

    for (int x = end_x; x >= min_x && x <= max_x; x -= step_x) {
      vector<bool> not_start(3);
      not_start[0] = x != start_x;
      not_start[1] = y != start_y;
      not_start[2] = true;

      vector<bool> not_end(3);
      not_end[0] = x != end_x;
      not_end[1] = y != end_y;
      not_end[2] = false;

      const Dtype *top_hidden_diff = top[1]->cpu_diff() + top[1]->offset(y, x);

      Dtype *H_diff_ptr = H_diff + H_diff_.offset(y, x);

      for (int dim = 0; dim < 3; ++dim) {
        int blob_offset = dim * num_blobs_per_dimension_;

        const Dtype *param_W_i_data = this->blobs_[blob_offset]->cpu_data();
        const Dtype *param_W_c_data =
            this->blobs_[blob_offset + 1]->cpu_data();
        const Dtype *param_W_o_data =
            this->blobs_[blob_offset + 2]->cpu_data();
        const Dtype *param_W_f_data =
            this->blobs_[blob_offset + 3]->cpu_data();
        const Dtype *param_W_i_c_data =
            this->blobs_[blob_offset + 4]->cpu_data();
        const Dtype *param_W_o_c_data =
            this->blobs_[blob_offset + 5]->cpu_data();
        const Dtype *param_W_f_c_data =
            this->blobs_[blob_offset + 6]->cpu_data();

        Dtype *param_W_i_diff = this->blobs_[blob_offset]->mutable_cpu_diff();
        Dtype *param_W_c_diff =
            this->blobs_[blob_offset + 1]->mutable_cpu_diff();
        Dtype *param_W_o_diff =
            this->blobs_[blob_offset + 2]->mutable_cpu_diff();
        Dtype *param_W_f_diff =
            this->blobs_[blob_offset + 3]->mutable_cpu_diff();
        Dtype *param_W_i_c_diff =
            this->blobs_[blob_offset + 4]->mutable_cpu_diff();
        Dtype *param_W_o_c_diff =
            this->blobs_[blob_offset + 5]->mutable_cpu_diff();
        Dtype *param_W_f_c_diff =
            this->blobs_[blob_offset + 6]->mutable_cpu_diff();
        Dtype *bias_b_i_diff =
            this->blobs_[blob_offset + 7]->mutable_cpu_diff();
        Dtype *bias_b_c_diff =
            this->blobs_[blob_offset + 8]->mutable_cpu_diff();
        Dtype *bias_b_o_diff =
            this->blobs_[blob_offset + 9]->mutable_cpu_diff();
        Dtype *bias_b_f_diff =
            this->blobs_[blob_offset + 10]->mutable_cpu_diff();

        Dtype *hidden_diff = hidden_diff_[dim]->mutable_cpu_data();

        // fill hidden_diff
        int index = 0;
        for (int n = 0; n < num_; ++n) {
          const Dtype *H_next_diff = NULL;
          if (not_end[dim]) {
            if (dim == 0) {
              H_next_diff = H_diff_.cpu_data()
                  + H_diff_.offset(y, x + step_x, n);
            } else {
              H_next_diff = H_diff_.cpu_data()
                  + H_diff_.offset(y + step_y, x, n);
            }
          }

          for (int d = 0; d < num_output_; ++d) {
            if (dim == 2) {
              hidden_diff[index] = top_hidden_diff[index];
            } else {
              hidden_diff[index] = 0;
            }
            if (not_end[dim]) {
              hidden_diff[index] += H_next_diff[dim * num_output_ + d];
            }
            index++;
          }
        }

        const Dtype *gi_data = gi_data_[dim]->cpu_data()
            + gi_data_[dim]->offset(y, x);
        Dtype *gi_diff = gi_same_row_diff_[dim]->mutable_cpu_data()
            + gi_same_row_diff_[dim]->offset(x);
        const Dtype *ci_data = ci_data_[dim]->cpu_data()
            + ci_data_[dim]->offset(y, x);
        Dtype *ci_diff = ci_diff_[dim]->mutable_cpu_data();
        const Dtype *go_data = go_data_[dim]->cpu_data()
            + go_data_[dim]->offset(y, x);
        Dtype *go_diff = go_diff_[dim]->mutable_cpu_data();
        const Dtype *gf_data = gf_data_[dim]->cpu_data()
            + gf_data_[dim]->offset(y, x);
        Dtype *gf_diff = gf_same_row_diff_[dim]->mutable_cpu_data()
            + gf_same_row_diff_[dim]->offset(x);
        const Dtype *cstate_data = cstate_data_[dim]->cpu_data()
            + cstate_data_[dim]->offset(y, x);
        Dtype *cstate_diff = cstate_same_row_diff_[dim]->mutable_cpu_data()
            + cstate_same_row_diff_[dim]->offset(x);

        const Dtype *gf_next_data = NULL;
        if (not_end[dim]) {
          if (dim == 0) {
            gf_next_data = gf_data_[dim]->cpu_data()
                + gf_data_[dim]->offset(y, x + step_x);
          } else {
            gf_next_data = gf_data_[dim]->cpu_data()
                + gf_data_[dim]->offset(y + step_y, x);
          }
        }

        const Dtype *cstate_prev_data = NULL;
        if (not_start[dim]) {
          switch (dim) {
          case 0: {
            cstate_prev_data = cstate_data_[dim]->cpu_data()
                + cstate_data_[dim]->offset(y, x - step_x);
            break;
          }
          case 1: {
            cstate_prev_data = cstate_data_[dim]->cpu_data()
                + cstate_data_[dim]->offset(y - step_y, x);
            break;
          }
          case 2: {
            cstate_prev_data = bottom[0]->cpu_data() + bottom[0]->offset(y, x);
            break;
          }
          }
        }

        const Dtype *cstate_next_diff = NULL;
        Dtype *gi_next_diff = NULL;
        const Dtype *gf_next_diff = NULL;
        if (not_end[dim]) {
          if (dim == 0) {
            cstate_next_diff = cstate_same_row_diff_[dim]->cpu_data()
                + cstate_same_row_diff_[dim]->offset(x + step_x);
            gi_next_diff = gi_same_row_diff_[dim]->mutable_cpu_data()
                + gi_same_row_diff_[dim]->offset(x + step_x);
            gf_next_diff = gf_same_row_diff_[dim]->cpu_data()
                + gf_same_row_diff_[dim]->offset(x + step_x);
          } else {
            cstate_next_diff = cstate_next_row_diff_[dim]->cpu_data()
                + cstate_next_row_diff_[dim]->offset(x);
            gi_next_diff = gi_next_row_diff_[dim]->mutable_cpu_data()
                + gi_next_row_diff_[dim]->offset(x);
            gf_next_diff = gf_next_row_diff_[dim]->cpu_data()
                + gf_next_row_diff_[dim]->offset(x);
          }
        }

        index = 0;
        for (int n = 0; n < num_; ++n) {
          for (int d = 0; d < num_output_; ++d) {
            Dtype cstate_val = cstate_data[index];
            Dtype go_val = go_data[index];
            go_diff[index] = hidden_diff[index] * tanh<Dtype>(cstate_val)
                * sigmoid_diff_y<Dtype>(go_val);
            cstate_diff[index] += hidden_diff[index] * go_val
                * tanh_diff_x<Dtype>(cstate_val);
            if (not_end[dim]) {
              cstate_diff[index] += cstate_next_diff[index]
                  * gf_next_data[index];
            }
            index++;
          }
        }

        if (peephole_) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
              num_output_, (Dtype) 1., go_diff, param_W_o_c_data, (Dtype) 1.,
              cstate_diff);
          if (not_end[dim]) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
                num_output_, num_output_, (Dtype) 1., gf_next_diff,
                param_W_f_c_data, (Dtype) 1., cstate_diff);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
                num_output_, num_output_, (Dtype) 1., gi_next_diff,
                param_W_i_c_data, (Dtype) 1., cstate_diff);
          }
        }

        index = 0;
        for (int n = 0; n < num_; ++n) {
          for (int d = 0; d < num_output_; ++d) {
            if (not_start[dim]) {
              gf_diff[index] = cstate_diff[index] * cstate_prev_data[index]
                  * sigmoid_diff_y<Dtype>(gf_data[index]);
            } else {
              gf_diff[index] = 0;
            }
            Dtype gi_val = gi_data[index];
            Dtype ci_val = ci_data[index];
            gi_diff[index] = cstate_diff[index] * ci_val
                * sigmoid_diff_y<Dtype>(gi_val);
            ci_diff[index] = cstate_diff[index] * gi_val
                * tanh_diff_y<Dtype>(ci_val);
            index++;
          }
        }

        // compute gradients w.r.t H=[H^x H^y H^z]
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
            3 * num_output_, num_output_, (Dtype) 1., gi_diff, param_W_i_data,
            (Dtype) 1., H_diff_ptr);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
            3 * num_output_, num_output_, (Dtype) 1., ci_diff, param_W_c_data,
            (Dtype) 1., H_diff_ptr);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
            3 * num_output_, num_output_, (Dtype) 1., go_diff, param_W_o_data,
            (Dtype) 1., H_diff_ptr);
        if (not_start[dim]) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_,
              3 * num_output_, num_output_, (Dtype) 1., gf_diff,
              param_W_f_data, (Dtype) 1., H_diff_ptr);
        }

        // compute gradients w.r.t. layer parameter matrices
        const Dtype *H_data_ptr = H_data + H_data_.offset(y, x);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            3 * num_output_, num_, (Dtype) 1., gi_diff, H_data_ptr, (Dtype) 1.,
            param_W_i_diff);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            3 * num_output_, num_, (Dtype) 1., ci_diff, H_data_ptr, (Dtype) 1.,
            param_W_c_diff);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            3 * num_output_, num_, (Dtype) 1., go_diff, H_data_ptr, (Dtype) 1.,
            param_W_o_diff);
        if (not_start[dim]) {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
              3 * num_output_, num_, (Dtype) 1., gf_diff, H_data_ptr,
              (Dtype) 1., param_W_f_diff);
          if (peephole_) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
                num_output_, num_, (Dtype) 1., gi_diff, cstate_prev_data,
                (Dtype) 1., param_W_i_c_diff);
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
                num_output_, num_, (Dtype) 1., gf_diff, cstate_prev_data,
                (Dtype) 1., param_W_f_c_diff);
          }
        }
        if (peephole_) {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
              num_output_, num_, (Dtype) 1., go_diff, cstate_data, (Dtype) 1.,
              param_W_o_c_diff);
        }

        // compute gradients w.r.t. layer biases
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
            gi_diff, bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_i_diff);
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
            ci_diff, bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_c_diff);
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
            go_diff, bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_o_diff);
        if (not_start[dim]) {
          caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
              gf_diff, bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_f_diff);
        }
      }  // for (int dim = 0; dim < 3; ++dim)

      // copy gradients w.r.t H into bottom[1] diff
      int index = 0;
      Dtype *bottom_hidden_diff = bottom[1]->mutable_cpu_diff()
          + bottom[1]->offset(y, x);
      for (int n = 0; n < num_; ++n) {
        H_diff_ptr = H_diff + H_diff_.offset(y, x, n, 2 * num_output_);
        for (int d = 0; d < num_output_; ++d) {
          bottom_hidden_diff[index] = H_diff_ptr[d];
          index++;
        }
      }

      // compute gradients w.r.t. bottom[0] (a.k.a previous cell state)
      Dtype *bottom_cstate_diff = bottom[0]->mutable_cpu_diff()
          + bottom[0]->offset(y, x);
      const Dtype *gf_data = gf_data_[2]->cpu_data()
          + gf_data_[2]->offset(y, x);
      const Dtype *cstate_diff = cstate_same_row_diff_[2]->cpu_data()
          + cstate_same_row_diff_[2]->offset(x);

      index = 0;
      for (int n = 0; n < num_; ++n) {
        for (int d = 0; d < num_output_; ++d) {
          bottom_cstate_diff[index] = gf_data[index] * cstate_diff[index];
          index++;
        }
      }

      if (peephole_) {
        int blob_offset = 2 * num_blobs_per_dimension_;
        const Dtype *param_W_i_c_data =
            this->blobs_[blob_offset + 4]->cpu_data();
        const Dtype *param_W_f_c_data =
            this->blobs_[blob_offset + 6]->cpu_data();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
            num_output_, (Dtype) 1.,
            gf_same_row_diff_[2]->cpu_data() + gf_same_row_diff_[2]->offset(x),
            param_W_f_c_data, (Dtype) 1., bottom_cstate_diff);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
            num_output_, (Dtype) 1.,
            gi_same_row_diff_[2]->cpu_data() + gi_same_row_diff_[2]->offset(x),
            param_W_i_c_data, (Dtype) 1., bottom_cstate_diff);
      }
    }  // for (int x = end_x; x >= min_x && x <= max_x; x -= step_x)
    for (int dim = 0; dim < 3; ++dim) {
      caffe_copy<Dtype>(gi_same_row_diff_[dim]->count(),
          gi_same_row_diff_[dim]->cpu_data(),
          gi_next_row_diff_[dim]->mutable_cpu_data());
      caffe_copy<Dtype>(gf_same_row_diff_[dim]->count(),
          gf_same_row_diff_[dim]->cpu_data(),
          gf_next_row_diff_[dim]->mutable_cpu_data());
      caffe_copy<Dtype>(cstate_same_row_diff_[dim]->count(),
          cstate_same_row_diff_[dim]->cpu_data(),
          cstate_next_row_diff_[dim]->mutable_cpu_data());
    }
  }  // for (int y = end_y; y >= min_y && y <= max_y; y -= step_y)
}

INSTANTIATE_CLASS(GridLSTM3DLayer);
REGISTER_LAYER_CLASS(GridLSTM3D);
}  // namespace caffe

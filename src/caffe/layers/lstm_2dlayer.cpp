#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/lstm_2dlayer.hpp"

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
LSTM_2DLayer_Forward_Worker<Dtype>::LSTM_2DLayer_Forward_Worker(
    int dir, LSTM_2DLayer<Dtype> *layer, Blob<Dtype>* bottom,
    Blob<Dtype>* top) :
    InternalThread(), dir_(dir), layer_(layer), bottom_(bottom), top_(
        top) {
}

/* Forward pass
 * input gate:i_{y,x}=sigmoid(W_{i,x}*X_{y,x}+H^x_i*X_{y,x-1}+
 *                    H^y_i*X_{y-1,x}+b_i)
 * cell input c_{y,x}=tanh(W_{c,x}*X_{y,x}+H^x_c*X_{y,x-1}+
 *                    H^y_c*X_{y-1,x}+b_c)
 * output gate:o_{y,x}=sigmoid(W_{o,x}*X_{y,x}+H^x_o*X_{y,x-1}+
 *                      H^y_o*X_{y-1,x}+b_o)
 * x forget gate:f^x_{y,x}=sigmoid(W^x_{f,x}*X_{y,x}+H^x_{f,x}*X_{y,x-1}
 *                          +H^y_{f,x}*X_{y-1,x}+b_{f,x})
 * y forget gate:f^y_{y,x}=sigmoid(W^x_{f,x}*X_{y,x}+H^x_{f,y}*X_{y,x-1}
 *                          +H^y_{f,y}*X_{y-1,x}+b_{f,y})
 * cell state s_{y,x}=c_{y,x}.*i_{y,x}+factor*f^x_{y,x}.*s_{y,x-1}
 *                    +factor*f^y_{y,x}.*s_{y-1,x}
 * */
template<typename Dtype>
void LSTM_2DLayer_Forward_Worker<Dtype>::InternalThreadEntry() {
  int y_dir = dir_ / 2;
  int x_dir = dir_ % 2;

  const int start_y = y_dir == 0 ? 0 : layer_->patch_ny_ - 1;
  const int end_y = y_dir == 0 ? layer_->patch_ny_ - 1 : 0;
  const int min_y = start_y <= end_y ? start_y : end_y;
  const int max_y = start_y <= end_y ? end_y : start_y;
  const int step_y = y_dir == 0 ? 1 : -1;

  const int start_x = x_dir == 0 ? 0 : layer_->patch_nx_ - 1;
  const int end_x = x_dir == 0 ? layer_->patch_nx_ - 1 : 0;
  const int min_x = start_x <= end_x ? start_x : end_x;
  const int max_x = start_x <= end_x ? end_x : start_x;
  const int step_x = x_dir == 0 ? 1 : -1;

  // W_i = [W_{i,x}, H^x_i, H^y_i]
  // W_c = [W_{c,x}, H^x_c, H^y_c]
  // W_o = [W_{o,x}, H^x_o, H^y_o]
  // W^x_f = [W^x_{f,x}, H^x_{f,x}, H^y_{f,x}]
  // W^y_f = [W^y_{f,x}, H^y_{f,x}, H^y_{f,y}]
  const Dtype *param_W_i_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_]->cpu_data();
  const Dtype *param_W_c_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype *param_W_o_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype *param_W_xf_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 3]->cpu_data();
  const Dtype *param_W_yf_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 4]->cpu_data();

  Dtype *X_Hx_Hy_data =
      layer_->X_Hx_Hy_data_[dir_]->mutable_cpu_data();
  Dtype *hidden_same_row_data =
      layer_->hidden_same_row_data_[dir_]->mutable_cpu_data();
  Dtype *hidden_prev_row_data =
      layer_->hidden_prev_row_data_[dir_]->mutable_cpu_data();

  const Dtype* bottom_data = bottom_->cpu_data();
  Dtype *top_data = top_->mutable_cpu_data();
  for (int y = start_y; y >= min_y && y <= max_y; y += step_y) {
    for (int x = start_x; x >= min_x && x <= max_x; x += step_x) {
      // fill X data
      for (int n = 0; n < layer_->num_; ++n) {
        Dtype* X_Hx_Hy_data_ptr = X_Hx_Hy_data
            + layer_->X_Hx_Hy_data_[dir_]->offset(y, x, n);
        int X_Hx_Hy_data_index = 0;
        for (int ch = 0; ch < layer_->channels_; ++ch) {
          for (int py = 0; py < layer_->patch_h_; ++py) {
            for (int px = 0; px < layer_->patch_w_; ++px) {
              X_Hx_Hy_data_ptr[X_Hx_Hy_data_index] =
                  bottom_data[bottom_->offset(n, ch,
                      y * layer_->patch_h_ + py,
                      x * layer_->patch_w_ + px)];
              X_Hx_Hy_data_index++;
            }
          }
        }
      }
      // fill previous hidden outputs in x and y directions
      for (int n = 0; n < layer_->num_; ++n) {
        Dtype* X_Hx_Hy_data_ptr = X_Hx_Hy_data
            + layer_->X_Hx_Hy_data_[dir_]->offset(y, x, n)
            + layer_->patch_dim_;
        int X_Hx_Hy_data_index = 0;
        for (int d = 0; d < layer_->num_output_; ++d) {
          if (x == start_x) {
            X_Hx_Hy_data_ptr[X_Hx_Hy_data_index++] = 0;
          } else {
            int offset = layer_->hidden_same_row_data_[dir_]->offset(
                x - step_x, n, d);
            X_Hx_Hy_data_ptr[X_Hx_Hy_data_index++] =
                hidden_same_row_data[offset];
          }
        }
      }

      for (int n = 0; n < layer_->num_; ++n) {
        Dtype* X_Hx_Hy_data_ptr = X_Hx_Hy_data
            + layer_->X_Hx_Hy_data_[dir_]->offset(y, x, n)
            + layer_->patch_dim_ + layer_->num_output_;
        int X_Hx_Hy_data_index = 0;
        for (int d = 0; d < layer_->num_output_; ++d) {
          if (y == start_y) {
            X_Hx_Hy_data_ptr[X_Hx_Hy_data_index++] = 0;
          } else {
            int offset = layer_->hidden_prev_row_data_[dir_]->offset(
                x, n, d);
            X_Hx_Hy_data_ptr[X_Hx_Hy_data_index++] =
                hidden_prev_row_data[offset];
          }
        }
      }

      Dtype* gi_data = layer_->gi_data_[dir_]->mutable_cpu_data()
          + layer_->gi_data_[dir_]->offset(y, x);
      Dtype* ci_data = layer_->ci_data_[dir_]->mutable_cpu_data()
          + layer_->ci_data_[dir_]->offset(y, x);
      Dtype* go_data = layer_->go_data_[dir_]->mutable_cpu_data()
          + layer_->go_data_[dir_]->offset(y, x);
      Dtype* gfx_data = layer_->gfx_data_[dir_]->mutable_cpu_data()
          + layer_->gfx_data_[dir_]->offset(y, x);
      Dtype* gfy_data = layer_->gfy_data_[dir_]->mutable_cpu_data()
          + layer_->gfy_data_[dir_]->offset(y, x);
      Dtype* cstate_data =
          layer_->cstate_data_[dir_]->mutable_cpu_data()
              + layer_->cstate_data_[dir_]->offset(y, x);
      Dtype* hidden_data = hidden_same_row_data
          + layer_->hidden_same_row_data_[dir_]->offset(x);

      Dtype* X_Hx_Hy_data_ptr = X_Hx_Hy_data
          + layer_->X_Hx_Hy_data_[dir_]->offset(y, x);
      // compute cell data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, layer_->num_,
          layer_->num_output_,
          (layer_->patch_dim_ + 2 * layer_->num_output_), (Dtype) 1.,
          X_Hx_Hy_data_ptr, param_W_i_data, (Dtype) 0., gi_data);
      // add bias
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          layer_->num_output_, 1, (Dtype) 1.,
          layer_->bias_multiplier_.cpu_data(),
          layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + 5]->cpu_data(),
          (Dtype) 1., gi_data);

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, layer_->num_,
          layer_->num_output_,
          (layer_->patch_dim_ + 2 * layer_->num_output_), (Dtype) 1.,
          X_Hx_Hy_data_ptr, param_W_c_data, (Dtype) 0., ci_data);
      // add bias
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          layer_->num_output_, 1, (Dtype) 1.,
          layer_->bias_multiplier_.cpu_data(),
          layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + 6]->cpu_data(),
          (Dtype) 1., ci_data);

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, layer_->num_,
          layer_->num_output_,
          (layer_->patch_dim_ + 2 * layer_->num_output_), (Dtype) 1.,
          X_Hx_Hy_data_ptr, param_W_o_data, (Dtype) 0., go_data);
      // add bias
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          layer_->num_output_, 1, (Dtype) 1.,
          layer_->bias_multiplier_.cpu_data(),
          layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + 7]->cpu_data(),
          (Dtype) 1., go_data);

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, layer_->num_,
          layer_->num_output_,
          (layer_->patch_dim_ + 2 * layer_->num_output_), (Dtype) 1.,
          X_Hx_Hy_data_ptr, param_W_xf_data, (Dtype) 0., gfx_data);
      // add bias
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          layer_->num_output_, 1, (Dtype) 1.,
          layer_->bias_multiplier_.cpu_data(),
          layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + 8]->cpu_data(),
          (Dtype) 1., gfx_data);

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, layer_->num_,
          layer_->num_output_,
          (layer_->patch_dim_ + 2 * layer_->num_output_), (Dtype) 1.,
          X_Hx_Hy_data_ptr, param_W_yf_data, (Dtype) 0., gfy_data);
      // add bias
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          layer_->num_output_, 1, (Dtype) 1.,
          layer_->bias_multiplier_.cpu_data(),
          layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + 9]->cpu_data(),
          (Dtype) 1., gfy_data);

      const Dtype* cstate_data_x_prev_ptr = NULL;
      if (x != start_x) {
        cstate_data_x_prev_ptr =
            layer_->cstate_data_[dir_]->mutable_cpu_data()
                + layer_->cstate_data_[dir_]->offset(y, x - step_x);
      }
      const Dtype* cstate_data_y_prev_ptr = NULL;
      if (y != start_y) {
        cstate_data_y_prev_ptr =
            layer_->cstate_data_[dir_]->mutable_cpu_data()
                + layer_->cstate_data_[dir_]->offset(y - step_y, x);
      }

      int data_index = 0;
      for (int n = 0; n < layer_->num_; ++n) {
        for (int d = 0; d < layer_->num_output_; ++d) {
          gi_data[data_index] = sigmoid<Dtype>(gi_data[data_index]);
          ci_data[data_index] = tanh<Dtype>(ci_data[data_index]);
          go_data[data_index] = sigmoid<Dtype>(go_data[data_index]);
          gfx_data[data_index] = sigmoid<Dtype>(gfx_data[data_index]);
          gfy_data[data_index] = sigmoid<Dtype>(gfy_data[data_index]);

          cstate_data[data_index] = ci_data[data_index]
              * gi_data[data_index];
          if (x != start_x) {
            cstate_data[data_index] +=
                layer_->forget_gate_scaling_factor_
                    * gfx_data[data_index]
                    * cstate_data_x_prev_ptr[data_index];
          }
          if (y != start_y) {
            cstate_data[data_index] +=
                layer_->forget_gate_scaling_factor_
                    * gfy_data[data_index]
                    * cstate_data_y_prev_ptr[data_index];
          }
          hidden_data[data_index] = go_data[data_index]
              * tanh<Dtype>(cstate_data[data_index]);
          // copy hidden output into top blob
          top_data[top_->offset(n, dir_ * layer_->num_output_ + d, y,
              x)] = hidden_data[data_index];
          data_index++;
        }
      }
    }  // for (int x = start_x; x >= min_x && x <= max_x; x += step_x)
    caffe_copy<Dtype>(layer_->hidden_same_row_data_[dir_]->count(),
        layer_->hidden_same_row_data_[dir_]->cpu_data(),
        layer_->hidden_prev_row_data_[dir_]->mutable_cpu_data());
  }  // for (int y = start_y; y >= min_y && y <= max_y; y += step_y)
}

INSTANTIATE_CLASS(LSTM_2DLayer_Forward_Worker);

template<typename Dtype>
LSTM_2DLayer_Backward_Worker<Dtype>::LSTM_2DLayer_Backward_Worker(
    int dir, LSTM_2DLayer<Dtype> *layer, Blob<Dtype>* bottom,
    Blob<Dtype>* top) :
    InternalThread(), dir_(dir), layer_(layer), bottom_(bottom), top_(
        top) {
  bottom_diff_.reset(new Blob<Dtype>());
}

template<typename Dtype>
void LSTM_2DLayer_Backward_Worker<Dtype>::InternalThreadEntry() {
  bottom_diff_->ReshapeLike(*bottom_);
  Dtype *bottom_diff = bottom_diff_->mutable_cpu_data();
  caffe_set<Dtype>(bottom_diff_->count(), 0, bottom_diff);

  int y_dir = dir_ / 2;
  int x_dir = dir_ % 2;

  const int start_y = y_dir == 0 ? 0 : layer_->patch_ny_ - 1;
  const int end_y = y_dir == 0 ? layer_->patch_ny_ - 1 : 0;
  const int min_y = start_y <= end_y ? start_y : end_y;
  const int max_y = start_y <= end_y ? end_y : start_y;
  const int step_y = y_dir == 0 ? 1 : -1;

  const int start_x = x_dir == 0 ? 0 : layer_->patch_nx_ - 1;
  const int end_x = x_dir == 0 ? layer_->patch_nx_ - 1 : 0;
  const int min_x = start_x <= end_x ? start_x : end_x;
  const int max_x = start_x <= end_x ? end_x : start_x;
  const int step_x = x_dir == 0 ? 1 : -1;

  for (int i = 0; i < layer_->num_blobs_per_dir_; ++i) {
    Dtype* param_diff = layer_->blobs_[dir_
        * layer_->num_blobs_per_dir_ + i]->mutable_cpu_diff();
    caffe_set<Dtype>(
        layer_->blobs_[dir_ * layer_->num_blobs_per_dir_ + i]->count(),
        0, param_diff);
  }

  const Dtype *param_W_i_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_]->cpu_data();
  const Dtype *param_W_c_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype *param_W_o_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype *param_W_xf_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 3]->cpu_data();
  const Dtype *param_W_yf_data = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 4]->cpu_data();

  Dtype *param_W_i_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_]->mutable_cpu_diff();
  Dtype *param_W_c_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 1]->mutable_cpu_diff();
  Dtype *param_W_o_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 2]->mutable_cpu_diff();
  Dtype *param_W_xf_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 3]->mutable_cpu_diff();
  Dtype *param_W_yf_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 4]->mutable_cpu_diff();
  Dtype* bias_b_i_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 5]->mutable_cpu_diff();
  Dtype* bias_b_c_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 6]->mutable_cpu_diff();
  Dtype* bias_b_o_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 7]->mutable_cpu_diff();
  Dtype* bias_b_fx_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 8]->mutable_cpu_diff();
  Dtype* bias_b_fy_diff = layer_->blobs_[dir_
      * layer_->num_blobs_per_dir_ + 9]->mutable_cpu_diff();

  const Dtype *X_Hx_Hy_data = layer_->X_Hx_Hy_data_[dir_]->cpu_data();
  Dtype *X_Hx_Hy_same_row_diff =
      layer_->X_Hx_Hy_same_row_diff_[dir_]->mutable_cpu_data();
  Dtype *X_Hx_Hy_next_row_diff =
      layer_->X_Hx_Hy_next_row_diff_[dir_]->mutable_cpu_data();
  Dtype *hidden_diff = layer_->hidden_diff_[dir_]->mutable_cpu_data();

  const Dtype *top_diff = top_->cpu_diff();
  const int X_Hx_Hy_dim = layer_->patch_dim_
      + 2 * layer_->num_output_;

  for (int y = end_y; y >= min_y && y <= max_y; y -= step_y) {
    for (int x = end_x; x >= min_x && x <= max_x; x -= step_x) {
      // copy top diff into hidden_diff
      int data_index = 0;
      for (int n = 0; n < layer_->num_; ++n) {
        for (int d = 0; d < layer_->num_output_; ++d) {
          hidden_diff[data_index] = top_diff[top_->offset(n,
              dir_ * layer_->num_output_ + d, y, x)];
          if (x != end_x) {
            Dtype* X_Hx_Hy_diff_next_x_ptr = X_Hx_Hy_same_row_diff
                + layer_->X_Hx_Hy_same_row_diff_[dir_]->offset(
                    x + step_x, n);
            hidden_diff[data_index] +=
                X_Hx_Hy_diff_next_x_ptr[layer_->patch_dim_ + d];
          }
          if (y != end_y) {
            Dtype* X_Hx_Hy_diff_next_y_ptr = X_Hx_Hy_next_row_diff
                + layer_->X_Hx_Hy_next_row_diff_[dir_]->offset(x, n);
            hidden_diff[data_index] +=
                X_Hx_Hy_diff_next_y_ptr[layer_->patch_dim_
                    + layer_->num_output_ + d];
          }
          data_index++;
        }
      }

      const Dtype* gi_data = layer_->gi_data_[dir_]->cpu_data()
          + layer_->gi_data_[dir_]->offset(y, x);
      Dtype* gi_diff = layer_->gi_diff_[dir_]->mutable_cpu_data();

      const Dtype* ci_data = layer_->ci_data_[dir_]->cpu_data()
          + layer_->ci_data_[dir_]->offset(y, x);
      Dtype* ci_diff = layer_->ci_diff_[dir_]->mutable_cpu_data();

      const Dtype* go_data = layer_->go_data_[dir_]->cpu_data()
          + layer_->go_data_[dir_]->offset(y, x);
      Dtype* go_diff = layer_->go_diff_[dir_]->mutable_cpu_data();

      const Dtype* gfx_data = layer_->gfx_data_[dir_]->cpu_data()
          + layer_->gfx_data_[dir_]->offset(y, x);
      Dtype* gfx_diff = layer_->gfx_diff_[dir_]->mutable_cpu_data();

      const Dtype* gfy_data = layer_->gfy_data_[dir_]->cpu_data()
          + layer_->gfy_data_[dir_]->offset(y, x);
      Dtype* gfy_diff = layer_->gfy_diff_[dir_]->mutable_cpu_data();

      const Dtype* cstate_data =
          layer_->cstate_data_[dir_]->cpu_data()
              + layer_->cstate_data_[dir_]->offset(y, x);
      Dtype* cstate_diff =
          layer_->cstate_same_row_diff_[dir_]->mutable_cpu_data()
              + layer_->cstate_same_row_diff_[dir_]->offset(x);

      const Dtype* cstate_next_x_diff = NULL;
      const Dtype* gfx_next_x_data = NULL;
      if (x != end_x) {
        cstate_next_x_diff =
            layer_->cstate_same_row_diff_[dir_]->cpu_data()
                + layer_->cstate_same_row_diff_[dir_]->offset(
                    x + step_x);
        gfx_next_x_data = layer_->gfx_data_[dir_]->cpu_data()
            + layer_->gfx_data_[dir_]->offset(y, x + step_x);
      }
      const Dtype* cstate_next_y_diff = NULL;
      const Dtype* gfy_next_y_data = NULL;
      if (y != end_y) {
        cstate_next_y_diff =
            layer_->cstate_next_row_diff_[dir_]->cpu_data()
                + layer_->cstate_next_row_diff_[dir_]->offset(x);
        gfy_next_y_data = layer_->gfy_data_[dir_]->cpu_data()
            + layer_->gfy_data_[dir_]->offset(y + step_y, x);
      }

      const Dtype* cstate_prev_x_data = NULL;
      if (x != start_x) {
        cstate_prev_x_data = layer_->cstate_data_[dir_]->cpu_data()
            + layer_->cstate_data_[dir_]->offset(y, x - step_x);
      }

      const Dtype* cstate_prev_y_data = NULL;
      if (y != start_y) {
        cstate_prev_y_data = layer_->cstate_data_[dir_]->cpu_data()
            + layer_->cstate_data_[dir_]->offset(y - step_y, x);
      }

      data_index = 0;
      for (int n = 0; n < layer_->num_; ++n) {
        for (int d = 0; d < layer_->num_output_; ++d) {
          const Dtype cstate_val = cstate_data[data_index];
          const Dtype go_val = go_data[data_index];

          go_diff[data_index] = hidden_diff[data_index]
              * tanh<Dtype>(cstate_val)
              * sigmoid_diff_y<Dtype>(go_val);
          cstate_diff[data_index] = hidden_diff[data_index] * go_val
              * tanh_diff_x<Dtype>(cstate_val);
          if (x != end_x) {
            cstate_diff[data_index] += cstate_next_x_diff[data_index]
                * layer_->forget_gate_scaling_factor_
                * gfx_next_x_data[data_index];
          }
          if (y != end_y) {
            cstate_diff[data_index] += cstate_next_y_diff[data_index]
                * layer_->forget_gate_scaling_factor_
                * gfy_next_y_data[data_index];
          }
          if (x != start_x) {
            const Dtype fx_val = gfx_data[data_index];
            gfx_diff[data_index] = cstate_diff[data_index]
                * layer_->forget_gate_scaling_factor_
                * cstate_prev_x_data[data_index]
                * sigmoid_diff_y<Dtype>(fx_val);
          } else {
            gfx_diff[data_index] = 0;
          }
          if (y != start_y) {
            const Dtype fy_val = gfy_data[data_index];
            gfy_diff[data_index] = cstate_diff[data_index]
                * layer_->forget_gate_scaling_factor_
                * cstate_prev_y_data[data_index]
                * sigmoid_diff_y<Dtype>(fy_val);
          } else {
            gfy_diff[data_index] = 0;
          }
          const Dtype gi_val = gi_data[data_index];
          const Dtype ci_val = ci_data[data_index];
          gi_diff[data_index] = cstate_diff[data_index] * ci_val
              * sigmoid_diff_y<Dtype>(gi_val);
          ci_diff[data_index] = cstate_diff[data_index] * gi_val
              * tanh_diff_y<Dtype>(ci_val);
          data_index++;
        }
      }

      // compute gradients w.r.t. X_Hx_Hy_
      Dtype* X_Hx_Hy_diff = X_Hx_Hy_same_row_diff
          + layer_->X_Hx_Hy_same_row_diff_[dir_]->offset(x);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          X_Hx_Hy_dim, layer_->num_output_, (Dtype) 1., gi_diff,
          param_W_i_data, (Dtype) 0., X_Hx_Hy_diff);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          X_Hx_Hy_dim, layer_->num_output_, (Dtype) 1., ci_diff,
          param_W_c_data, (Dtype) 1., X_Hx_Hy_diff);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, layer_->num_,
          X_Hx_Hy_dim, layer_->num_output_, (Dtype) 1., go_diff,
          param_W_o_data, (Dtype) 1., X_Hx_Hy_diff);
      if (x != start_x) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            layer_->num_, X_Hx_Hy_dim, layer_->num_output_,
            (Dtype) 1., gfx_diff, param_W_xf_data, (Dtype) 1.,
            X_Hx_Hy_diff);
      }
      if (y != start_y) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            layer_->num_, X_Hx_Hy_dim, layer_->num_output_,
            (Dtype) 1., gfy_diff, param_W_yf_data, (Dtype) 1.,
            X_Hx_Hy_diff);
      }

      // copy gradients w.r.t. X_Hx_Hy into bottom diff
      for (int n = 0; n < layer_->num_; ++n) {
        X_Hx_Hy_diff = X_Hx_Hy_same_row_diff
            + layer_->X_Hx_Hy_same_row_diff_[dir_]->offset(x, n);
        data_index = 0;
        for (int ch = 0; ch < layer_->channels_; ++ch) {
          for (int py = 0; py < layer_->patch_h_; ++py) {
            for (int px = 0; px < layer_->patch_w_; ++px) {
              bottom_diff[bottom_diff_->offset(n, ch,
                  y * layer_->patch_h_ + py,
                  x * layer_->patch_w_ + px)] +=
                  X_Hx_Hy_diff[data_index++];
            }
          }
        }
      }
      // compute gradients w.r.t. parameters
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          layer_->num_output_, X_Hx_Hy_dim, layer_->num_, (Dtype) 1.,
          gi_diff,
          X_Hx_Hy_data + layer_->X_Hx_Hy_data_[dir_]->offset(y, x),
          (Dtype) 1., param_W_i_diff);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          layer_->num_output_, X_Hx_Hy_dim, layer_->num_, (Dtype) 1.,
          ci_diff,
          X_Hx_Hy_data + layer_->X_Hx_Hy_data_[dir_]->offset(y, x),
          (Dtype) 1., param_W_c_diff);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          layer_->num_output_, X_Hx_Hy_dim, layer_->num_, (Dtype) 1.,
          go_diff,
          X_Hx_Hy_data + layer_->X_Hx_Hy_data_[dir_]->offset(y, x),
          (Dtype) 1., param_W_o_diff);
      if (x != start_x) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            layer_->num_output_, X_Hx_Hy_dim, layer_->num_,
            (Dtype) 1., gfx_diff,
            X_Hx_Hy_data + layer_->X_Hx_Hy_data_[dir_]->offset(y, x),
            (Dtype) 1., param_W_xf_diff);
      }
      if (y != start_y) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            layer_->num_output_, X_Hx_Hy_dim, layer_->num_,
            (Dtype) 1., gfy_diff,
            X_Hx_Hy_data + layer_->X_Hx_Hy_data_[dir_]->offset(y, x),
            (Dtype) 1., param_W_yf_diff);
      }
      // compute gradients w.r.t. biases
      caffe_cpu_gemv<Dtype>(CblasTrans, layer_->num_,
          layer_->num_output_, (Dtype) 1., gi_diff,
          layer_->bias_multiplier_.cpu_data(), (Dtype) 1.,
          bias_b_i_diff);
      caffe_cpu_gemv<Dtype>(CblasTrans, layer_->num_,
          layer_->num_output_, (Dtype) 1., ci_diff,
          layer_->bias_multiplier_.cpu_data(), (Dtype) 1.,
          bias_b_c_diff);
      caffe_cpu_gemv<Dtype>(CblasTrans, layer_->num_,
          layer_->num_output_, (Dtype) 1., go_diff,
          layer_->bias_multiplier_.cpu_data(), (Dtype) 1.,
          bias_b_o_diff);
      if (x != start_x) {
        caffe_cpu_gemv<Dtype>(CblasTrans, layer_->num_,
            layer_->num_output_, (Dtype) 1., gfx_diff,
            layer_->bias_multiplier_.cpu_data(), (Dtype) 1.,
            bias_b_fx_diff);
      }
      if (y != start_y) {
        caffe_cpu_gemv<Dtype>(CblasTrans, layer_->num_,
            layer_->num_output_, (Dtype) 1., gfy_diff,
            layer_->bias_multiplier_.cpu_data(), (Dtype) 1.,
            bias_b_fy_diff);
      }
    }  // for (int x = end_x; x >= min_x && x <= max_x; x -= step_x)
    caffe_copy<Dtype>(layer_->X_Hx_Hy_same_row_diff_[dir_]->count(),
        layer_->X_Hx_Hy_same_row_diff_[dir_]->cpu_data(),
        layer_->X_Hx_Hy_next_row_diff_[dir_]->mutable_cpu_data());
    caffe_copy<Dtype>(layer_->cstate_same_row_diff_[dir_]->count(),
        layer_->cstate_same_row_diff_[dir_]->cpu_data(),
        layer_->cstate_next_row_diff_[dir_]->mutable_cpu_data());
  }  // for (int y = end_y; y >= min_y && y <= max_y; y -= step_y)
}

INSTANTIATE_CLASS(LSTM_2DLayer_Backward_Worker);

template<typename Dtype>
void LSTM_2DLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  num_output_ = this->layer_param_.lstm_2d_param().num_output();
  patch_h_ = this->layer_param_.lstm_2d_param().patch_height();
  patch_w_ = this->layer_param_.lstm_2d_param().patch_width();
  forget_gate_scaling_factor_ =
      this->layer_param_.lstm_2d_param().forget_gate_scaling_factor();

  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
  CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  patch_dim_ = channels_ * patch_h_ * patch_w_;

  X_Hx_Hy_data_.resize(4);
  X_Hx_Hy_same_row_diff_.resize(4);
  X_Hx_Hy_next_row_diff_.resize(4);
  gi_data_.resize(4);
  gi_diff_.resize(4);
  ci_data_.resize(4);
  ci_diff_.resize(4);
  go_data_.resize(4);
  go_diff_.resize(4);
  gfx_data_.resize(4);
  gfx_diff_.resize(4);
  gfy_data_.resize(4);
  gfy_diff_.resize(4);
  cstate_data_.resize(4);
  cstate_same_row_diff_.resize(4);
  cstate_next_row_diff_.resize(4);
  hidden_same_row_data_.resize(4);
  hidden_prev_row_data_.resize(4);
  hidden_diff_.resize(4);
  for (int dir = 0; dir < 4; dir++) {
    X_Hx_Hy_data_[dir].reset(new Blob<Dtype>());
    X_Hx_Hy_same_row_diff_[dir].reset(new Blob<Dtype>());
    X_Hx_Hy_next_row_diff_[dir].reset(new Blob<Dtype>());
    gi_data_[dir].reset(new Blob<Dtype>());
    gi_diff_[dir].reset(new Blob<Dtype>());
    ci_data_[dir].reset(new Blob<Dtype>());
    ci_diff_[dir].reset(new Blob<Dtype>());
    go_data_[dir].reset(new Blob<Dtype>());
    go_diff_[dir].reset(new Blob<Dtype>());
    gfx_data_[dir].reset(new Blob<Dtype>());
    gfx_diff_[dir].reset(new Blob<Dtype>());
    gfy_data_[dir].reset(new Blob<Dtype>());
    gfy_diff_[dir].reset(new Blob<Dtype>());
    cstate_data_[dir].reset(new Blob<Dtype>());
    cstate_same_row_diff_[dir].reset(new Blob<Dtype>());
    cstate_next_row_diff_[dir].reset(new Blob<Dtype>());
    hidden_same_row_data_[dir].reset(new Blob<Dtype>());
    hidden_prev_row_data_[dir].reset(new Blob<Dtype>());
    hidden_diff_[dir].reset(new Blob<Dtype>());
  }

  // 5 parameter matrices W_i, W_c, W_o, W^x_f and W^y_f
  // 5 bias vectors b_i, b_c, b_o, b^x_f and b^y_f
  num_blobs_per_dir_ = 10;
  this->blobs_.resize(4 * num_blobs_per_dir_);
  // W_i = [W_{i,x}, H^x_i, H^y_i]
  // W_c = [W_{c,x}, H^x_c, H^y_c]
  // W_o = [W_{o,x}, H^x_o, H^y_o]
  // W^x_f = [W^x_{f,x}, H^x_{f,x}, H^y_{f,x}]
  // W^y_f = [W^y_{f,x}, H^y_{f,x}, H^y_{f,y}]
  vector<int> W_shape(2);
  W_shape[0] = num_output_;
  W_shape[1] = patch_dim_ + num_output_ + num_output_;
  // bias blob shape
  vector<int> B_shape(1, num_output_);

  shared_ptr<Filler<Dtype> > general_weight_filler(
      GetFiller<Dtype>(
          this->layer_param_.lstm_2d_param().general_weight_filler()));
  shared_ptr<Filler<Dtype> > general_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.lstm_2d_param().general_bias_filler()));
  shared_ptr<Filler<Dtype> > forget_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.lstm_2d_param().forget_gate_bias_filler()));

  // 4 scanning directions
  for (int dir_c = 0; dir_c < 4; dir_c++) {
    // 5 parameter matrices W_i, W_c, W_o, W^x_f and W^y_f
    for (int p = 0; p < 5; ++p) {
      this->blobs_[dir_c * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(W_shape));
    }
    // 5 bias vectors b_i, b_c, b_o, b^x_f and b^y_f
    for (int p = 5; p < num_blobs_per_dir_; ++p) {
      this->blobs_[dir_c * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(B_shape));
    }
    // 5 parameter matrices W_i, W_c, W_o, W^x_f and W^y_f
    for (int p = 0; p < 5; ++p) {
      general_weight_filler->Fill(
          this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
    }
    // 3 bias vectors b_i, b_c, b_o
    for (int p = 5; p < 8; ++p) {
      general_bias_filler->Fill(
          this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
    }
    // 2 forget gate bias vectors b^x_f and b^y_f
    for (int p = 8; p < 10; ++p) {
      forget_gate_bias_filler->Fill(
          this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
    }
  }

  forward_workers_.resize(4);
  for (int dir = 0; dir < 4; ++dir) {
    forward_workers_[dir].reset(
        new LSTM_2DLayer_Forward_Worker<Dtype>(dir, this, bottom[0],
            top[0]));
  }

  backward_workers_.resize(4);
  for (int dir = 0; dir < 4; ++dir) {
    backward_workers_[dir].reset(
        new LSTM_2DLayer_Backward_Worker<Dtype>(dir, this, bottom[0],
            top[0]));
  }
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[0]->shape(1), channels_);
  CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
  CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

  num_ = bottom[0]->shape(0);
  patch_ny_ = bottom[0]->shape(2) / patch_h_;
  patch_nx_ = bottom[0]->shape(3) / patch_w_;

  // hold data for all image patches
  vector<int> X_Hx_Hy_data_shape(4);
  X_Hx_Hy_data_shape[0] = patch_ny_;
  X_Hx_Hy_data_shape[1] = patch_nx_;
  X_Hx_Hy_data_shape[2] = num_;
  X_Hx_Hy_data_shape[3] = patch_dim_ + num_output_ + num_output_;

  // hold gradients for a single row of image
  vector<int> X_Hx_Hy_diff_shape(3);
  X_Hx_Hy_diff_shape[0] = patch_nx_;
  X_Hx_Hy_diff_shape[1] = num_;
  X_Hx_Hy_diff_shape[2] = patch_dim_ + num_output_ + num_output_;

  // hold data for all image patches
  vector<int> cell_shape(4);
  cell_shape[0] = patch_ny_;
  cell_shape[1] = patch_nx_;
  cell_shape[2] = num_;
  cell_shape[3] = num_output_;

  // hold gradients for a single image patch
  vector<int> compact_cell_shape(2);
  compact_cell_shape[0] = num_;
  compact_cell_shape[1] = num_output_;

  // hold gradients for a single row of image
  vector<int> compact_row_cell_shape(3);
  compact_row_cell_shape[0] = patch_nx_;
  compact_row_cell_shape[1] = num_;
  compact_row_cell_shape[2] = num_output_;

  for (int i = 0; i < 4; ++i) {
    X_Hx_Hy_data_[i]->Reshape(X_Hx_Hy_data_shape);
    X_Hx_Hy_same_row_diff_[i]->Reshape(X_Hx_Hy_diff_shape);
    X_Hx_Hy_next_row_diff_[i]->Reshape(X_Hx_Hy_diff_shape);

    gi_data_[i]->Reshape(cell_shape);
    ci_data_[i]->Reshape(cell_shape);
    go_data_[i]->Reshape(cell_shape);
    gfx_data_[i]->Reshape(cell_shape);
    gfy_data_[i]->Reshape(cell_shape);

    gi_diff_[i]->Reshape(compact_cell_shape);
    ci_diff_[i]->Reshape(compact_cell_shape);
    go_diff_[i]->Reshape(compact_cell_shape);
    gfx_diff_[i]->Reshape(compact_cell_shape);
    gfy_diff_[i]->Reshape(compact_cell_shape);

    cstate_data_[i]->Reshape(cell_shape);
    cstate_same_row_diff_[i]->Reshape(compact_row_cell_shape);
    cstate_next_row_diff_[i]->Reshape(compact_row_cell_shape);

    hidden_same_row_data_[i]->Reshape(compact_row_cell_shape);
    hidden_prev_row_data_[i]->Reshape(compact_row_cell_shape);
    hidden_diff_[i]->Reshape(compact_cell_shape);
  }

  vector<int> top_shape(4);
  top_shape[0] = num_;
  top_shape[1] = 4 * num_output_;
  top_shape[2] = patch_ny_;
  top_shape[3] = patch_nx_;
  top[0]->Reshape(top_shape);

  vector<int> bias_shape(1, num_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set<Dtype>(num_, Dtype(1),
      bias_multiplier_.mutable_cpu_data());
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int dir = 0; dir < 4; ++dir) {
    CHECK(forward_workers_[dir]->StartInternalThread())<<
    "Thread execution failed";
  }
  for (int dir = 0; dir < 4; ++dir) {
    CHECK(forward_workers_[dir]->WaitForInternalThreadToExit()) <<
    "Thread joining failed";
  }
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(), 0, bottom_diff);

  for (int dir = 0; dir < 4; ++dir) {
    CHECK(backward_workers_[dir]->StartInternalThread())<<
    "Thread execution failed";
  }
  for (int dir = 0; dir < 4; ++dir) {
    CHECK(backward_workers_[dir]->WaitForInternalThreadToExit())<<
    "Thread joining failed";
  }
  for (int dir = 0; dir < 4; ++dir) {
    const Blob<Dtype>* bottom_diff_worker =
        backward_workers_[dir]->get_bottom_diff();
    const Dtype* bottom_diff_worker_data =
        bottom_diff_worker->cpu_data();
    for (int i = 0; i < bottom_diff_worker->count(); ++i) {
      bottom_diff[i] += bottom_diff_worker_data[i];
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(LSTM_2DLayer);
#endif

INSTANTIATE_CLASS(LSTM_2DLayer);
REGISTER_LAYER_CLASS(LSTM_2D);

} // namespace caffe

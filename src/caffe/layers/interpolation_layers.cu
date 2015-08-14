#include "caffe/interpolation_layers.hpp"
#include "caffe/common_math.cuh"

namespace caffe {

template<typename Dtype>
__global__ void bilinear_interpolation_forward(const int nthreads,
    const int img_num, const int channels, const int in_height,
    const int in_width, const int out_height, const int out_width,
    const Dtype* in_data, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (out_height * out_width);
    int rem = index % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    Dtype fy = (Dtype) (in_height - 1) * (Dtype) h / (Dtype) (out_height - 1);
    int prev_y = floorf(fy);
    int next_y = (prev_y + 1) == in_height ? prev_y : prev_y + 1;
    Dtype dy = fy - prev_y;

    Dtype fx = (Dtype) (in_width - 1) * (Dtype) w / (Dtype) (out_width - 1);
    int prev_x = floorf(fx);
    int next_x = (prev_x + 1) == in_width ? prev_x : prev_x + 1;
    Dtype dx = fx - prev_x;

    Dtype pp_weight = (1.0 - dy) * (1.0 - dx);
    Dtype pn_weight = (1.0 - dy) * dx;
    Dtype np_weight = dy * (1.0 - dx);
    Dtype nn_weight = dy * dx;

    const Dtype *in_data_ptr = in_data + n * channels * in_height * in_width;
    Dtype *out_data_ptr = out_data + n * channels * out_height * out_width;
    for (int c = 0; c < channels; ++c) {
      out_data_ptr[h * out_width + w] = pp_weight
          * in_data_ptr[prev_y * in_width + prev_x]
          + pn_weight * in_data_ptr[prev_y * in_width + next_x]
          + np_weight * in_data_ptr[next_y * in_width + prev_x]
          + nn_weight * in_data_ptr[next_y * in_width + next_x];

      in_data_ptr += in_height * in_width;
      out_data_ptr += out_height * out_width;
    }
  }
}

template<typename Dtype>
__global__ void bilinear_interpolation_backward(const int nthreads,
    const int img_num, const int channels, const int in_height,
    const int in_width, const int out_height, const int out_width,
    Dtype* in_diff, const Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (in_height * in_width);
    int rem = index % (in_height * in_width);
    int in_y = rem / in_width;
    int in_x = rem % in_width;

    int left_start = ceil((out_width - 1.0) * in_x / (in_width - 1.0));
    int left_end = ceil((out_width - 1.0) * (in_x + 1.0) / (in_width - 1.0))
        - 1;
    left_end = left_end < out_width ? left_end : out_width - 1;

    int right_start = ceil(
        (out_width - 1.0) * (in_x - 1.0) / (in_width - 1.0));
    right_start = right_start >= 0 ? right_start : 0;
    int right_end = ceil((out_width - 1.0) * (in_x) / (in_width - 1.0)) - 1;

    int top_start = ceil((out_height - 1.0) * in_y / (in_height - 1.0));
    int top_end = ceil((out_height - 1.0) * (in_y + 1.0) / (in_height - 1.0))
        - 1;
    top_end = top_end < out_height ? top_end : out_height - 1;

    int bottom_start = ceil(
        (out_height - 1.0) * (in_y - 1.0) / (in_height - 1.0));
    bottom_start = bottom_start >= 0 ? bottom_start : 0;
    int bottom_end = ceil((out_height - 1.0) * (in_y) / (in_height - 1.0)) - 1;

    // the current pixel is in the bottom-right corner
    for (int py = bottom_start; py <= bottom_end; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      int prev_y = floor(fy);
      Dtype dy = fy - prev_y;
      for (int px = right_start; px <= right_end; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        int prev_x = floor(fx);
        Dtype dx = fx - prev_x;
        for (int ch = 0; ch < channels; ++ch) {
          in_diff[blob_offset(channels, in_height, in_width, n, ch, in_y, in_x)]
                  += dy * dx
                  * out_diff[blob_offset(channels, out_height, out_width, n, ch,
                      py, px)];
        }
      }
    }

    // the current pixel is in the bottom-left corner
    for (int py = bottom_start; py <= bottom_end; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      int prev_y = floor(fy);
      Dtype dy = fy - prev_y;
      for (int px = left_start; px <= left_end; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        int prev_x = floor(fx);
        Dtype dx = fx - prev_x;
        for (int ch = 0; ch < channels; ++ch) {
          in_diff[blob_offset(channels, in_height, in_width, n, ch, in_y, in_x)]
                  += dy * (1.0 - dx)
                  * out_diff[blob_offset(channels, out_height, out_width, n, ch,
                      py, px)];
        }
      }
    }

    // the current pixel is in the top-right corner
    for (int py = top_start; py <= top_end; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      int prev_y = floor(fy);
      Dtype dy = fy - prev_y;
      for (int px = right_start; px <= right_end; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        int prev_x = floor(fx);
        Dtype dx = fx - prev_x;
        for (int ch = 0; ch < channels; ++ch) {
          in_diff[blob_offset(channels, in_height, in_width, n, ch, in_y, in_x)]
                  += (1.0 - dy) * dx
                  * out_diff[blob_offset(channels, out_height, out_width, n, ch,
                      py, px)];
        }
      }
    }

    // the current pixel is in the top-left corner
    for (int py = top_start; py <= top_end; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      int prev_y = floor(fy);
      Dtype dy = fy - prev_y;
      for (int px = left_start; px <= left_end; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        int prev_x = floor(fx);
        Dtype dx = fx - prev_x;
        for (int ch = 0; ch < channels; ++ch) {
          in_diff[blob_offset(channels, in_height, in_width, n, ch, in_y, in_x)]
                  += (1.0 - dy) * (1.0 - dx)
                  * out_diff[blob_offset(channels, out_height, out_width, n, ch,
                      py, px)];
        }
      }
    }
  }
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int ch = bottom[0]->shape(1);
  const int in_h = bottom[0]->shape(2);
  const int in_w = bottom[0]->shape(3);
  const int out_h = top[0]->shape(2);
  const int out_w = top[0]->shape(3);

  const int num_threads = num * out_h * out_w;
  bilinear_interpolation_forward<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(num_threads, num, ch, in_h, in_w, out_h, out_w,
      bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* in_diff_cpu = bottom[0]->mutable_cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(), 0, in_diff_cpu);

  const int num = bottom[0]->shape(0);
  const int ch = bottom[0]->shape(1);
  const int in_h = bottom[0]->shape(2);
  const int in_w = bottom[0]->shape(3);
  const int out_h = top[0]->shape(2);
  const int out_w = top[0]->shape(3);

  const int num_threads = num * in_h * in_w;
  bilinear_interpolation_backward<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(num_threads, num, ch, in_h, in_w, out_h, out_w,
      bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(BilinearInterpolationLayer);
}  // namespace caffe

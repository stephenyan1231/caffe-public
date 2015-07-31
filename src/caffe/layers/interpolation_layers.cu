#include "caffe/interpolation_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void bilinear_interpolation_forward(const int nthreads,
		const int img_num, const int img_channels, const int in_height,
		const int in_width, const int out_height, const int out_width,
		const Dtype* in_data, Dtype* out_data) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int n = index / (out_height * out_width);
		int rem = index % (out_height * out_width);
		int h = rem / out_width;
		int w = rem % out_width;

		Dtype fh = (Dtype) (in_height - 1) * (Dtype) h / (Dtype) (out_height - 1);
		int prev_h = floorf(fh);
		int next_h = (prev_h + 1) == in_height ? prev_h : prev_h + 1;
		Dtype dh = fh - prev_h;

		Dtype fw = (Dtype) (in_width - 1) * (Dtype) w / (Dtype) (out_width - 1);
		int prev_w = floorf(fw);
		int next_w = (prev_w + 1) == in_width ? prev_w : prev_w + 1;
		Dtype dw = fw - prev_w;

		Dtype pp_weight = (1.0 - dh) * (1.0 - dw);
		Dtype pn_weight = (1.0 - dh) * dw;
		Dtype np_weight = dh * (1.0 - dw);
		Dtype nn_weight = dh * dw;

		const Dtype* in_data_ptr = in_data
				+ n * img_channels * in_height * in_width;
		Dtype* out_data_ptr = out_data + n * img_channels * out_height * out_width;
		for (int c = 0; c < img_channels; ++c) {
			out_data_ptr[h * out_width + w] = pp_weight
					* in_data_ptr[prev_h * in_width + prev_w]
					+ pn_weight * in_data_ptr[prev_h * in_width + next_w]
					+ np_weight * in_data_ptr[next_h * in_width + prev_w]
					+ nn_weight * in_data_ptr[next_h * in_width + next_w];

			in_data_ptr += in_height * in_width;
			out_data_ptr += out_height * out_width;
		}
	}
}

// NOT WORKING NOW
template<typename Dtype>
__global__ void bilinear_interpolation_backward(const int nthreads,
    const int img_num, const int img_channels, const int in_height,
    const int in_width, const int out_height, const int out_width,
    Dtype* in_diff, const Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int n = index / (in_height * in_width);
    int rem = index % (in_height * in_width);
    int y = rem / in_width;
    int x = rem % in_width;

    Dtype* in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
    const Dtype* out_diff_ptr = out_diff
        + n * img_channels * out_height * out_width;
    const int in_diff_offset = y * in_width + x;

    int bottom_start_y = fmaxf(
        ceilf(y * (out_height - 1.0) / (in_height - 1.0)), 0);
    int bottom_end_y = fminf(
        ceilf((y + 1) * (out_height - 1.0) / (in_height - 1.0)) - 1,
        out_height - 1);

    int top_start_y = fmaxf(
        floorf((y - 1.0) * (out_height - 1.0) / (in_height - 1.0)) + 1, 0);
    int top_end_y = fminf(floorf(y * (out_height - 1.0) / (in_height - 1.0)),
        out_height - 1);

    int right_start_x = fmaxf(ceilf(x * (out_width - 1.0) / (in_width - 1.0)),
        0);
    int right_end_x = fminf(
        ceilf((x + 1) * (out_width - 1.0) / (in_width - 1.0)) - 1,
        out_width - 1);

    int left_start_x = fmaxf(
        floorf((x - 1.0) * (out_width - 1.0) / (in_width - 1.0)) + 1, 0);
    int left_end_x = fminf(floorf(y * (out_width - 1.0) / (in_width - 1.0)),
        out_width - 1);

    for (int py = top_start_y; py <= top_end_y; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      Dtype dy = fy - floorf(fy);
      for (int px = left_start_x; px <= left_end_x; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        Dtype dx = fx - floorf(fx);

        in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
        out_diff_ptr = out_diff + n * img_channels * out_height * out_width;
        const int out_diff_offset = py * out_width + px;
        for (int c = 0; c < img_channels; ++c) {
          in_diff_ptr[in_diff_offset] += out_diff_ptr[out_diff_offset] * dy
              * dx;
          in_diff_ptr += in_height * in_width;
          out_diff_ptr += out_height * out_width;
        }
      }
    }

    for (int py = top_start_y; py <= top_end_y; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      Dtype dy = fy - floorf(fy);
      for (int px = right_start_x; px <= right_end_x; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        Dtype dx = fx - floorf(fx);

        in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
        out_diff_ptr = out_diff + n * img_channels * out_height * out_width;
        const int out_diff_offset = py * out_width + px;
        for (int c = 0; c < img_channels; ++c) {
          in_diff_ptr[in_diff_offset] += out_diff_ptr[out_diff_offset] * dy
              * (1.0 - dx);
          in_diff_ptr += in_height * in_width;
          out_diff_ptr += out_height * out_width;
        }
      }
    }

    for (int py = bottom_start_y; py <= bottom_end_y; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      Dtype dy = fy - floorf(fy);
      for (int px = left_start_x; px <= left_end_x; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        Dtype dx = fx - floorf(fx);

        in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
        out_diff_ptr = out_diff + n * img_channels * out_height * out_width;
        const int out_diff_offset = py * out_width + px;
        for (int c = 0; c < img_channels; ++c) {
          in_diff_ptr[in_diff_offset] += out_diff_ptr[out_diff_offset]
              * (1.0 - dy) * dx;
          in_diff_ptr += in_height * in_width;
          out_diff_ptr += out_height * out_width;
        }
      }
    }

    for (int py = bottom_start_y; py <= bottom_end_y; ++py) {
      Dtype fy = (Dtype) (in_height - 1) * (Dtype) py
          / (Dtype) (out_height - 1);
      Dtype dy = fy - floorf(fy);
      for (int px = right_start_x; px <= right_end_x; ++px) {
        Dtype fx = (Dtype) (in_width - 1) * (Dtype) px
            / (Dtype) (out_width - 1);
        Dtype dx = fx - floorf(fx);

        in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
        out_diff_ptr = out_diff + n * img_channels * out_height * out_width;
        const int out_diff_offset = py * out_width + px;
        for (int c = 0; c < img_channels; ++c) {
          in_diff_ptr[in_diff_offset] += out_diff_ptr[out_diff_offset]
              * (1.0 - dy) * (1.0 - dx);
          in_diff_ptr += in_height * in_width;
          out_diff_ptr += out_height * out_width;
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
	CAFFE_CUDA_NUM_THREADS>>>(num_threads, num, ch, in_h, in_w,
			out_h, out_w, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);

//  Dtype* in_diff_cpu = bottom[0]->mutable_cpu_diff();
//  caffe_set<Dtype>(bottom[0]->count(), 0, in_diff_cpu);
//
//  const int num = bottom[0]->shape(0);
//  const int ch = bottom[0]->shape(1);
//  const int in_h = bottom[0]->shape(2);
//  const int in_w = bottom[0]->shape(3);
//  const int out_h = top[0]->shape(2);
//  const int out_w = top[0]->shape(3);
//
//  const int num_threads = num * in_h * in_w;
//  bilinear_interpolation_backward<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
//  CAFFE_CUDA_NUM_THREADS>>> (num_threads,
//      num, ch, in_h, in_w, out_h, out_w, bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(BilinearInterpolationLayer);

} // namespace caffe

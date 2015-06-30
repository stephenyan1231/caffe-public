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

template<typename Dtype>
__global__ void bilinear_interpolation_backward(const int nthreads,
		const int img_num, const int img_channels, const int in_height,
		const int in_width, const int out_height, const int out_width,
		Dtype* in_diff, const Dtype* out_diff) {
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

		Dtype* in_diff_ptr = in_diff + n * img_channels * in_height * in_width;
		const Dtype* out_diff_ptr = out_diff
				+ n * img_channels * out_height * out_width;
		for (int c = 0; c < img_channels; ++c) {
			Dtype out_diff_val = out_diff_ptr[h * out_width + w];
			in_diff_ptr[prev_h * in_width + prev_w] += pp_weight * out_diff_val;
			in_diff_ptr[prev_h * in_width + next_w] += pn_weight * out_diff_val;
			in_diff_ptr[next_h * in_width + prev_w] += np_weight * out_diff_val;
			in_diff_ptr[next_h * in_width + next_w] += nn_weight * out_diff_val;

			in_diff_ptr += in_height * in_width;
			out_diff_ptr += out_height * out_width;
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
	Dtype* in_diff = bottom[0]->mutable_cpu_diff();
	caffe_memset(bottom[0]->count() * sizeof(Dtype), 0, in_diff);

	const int num = bottom[0]->shape(0);
	const int ch = bottom[0]->shape(1);
	const int in_h = bottom[0]->shape(2);
	const int in_w = bottom[0]->shape(3);
	const int out_h = top[0]->shape(2);
	const int out_w = top[0]->shape(3);

	const int num_threads = num * out_h * out_w;
	bilinear_interpolation_backward<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
	CAFFE_CUDA_NUM_THREADS>>> (num_threads,
			num, ch, in_h, in_w, out_h, out_w, in_diff, top[0]->cpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(BilinearInterpolationLayer);

} // namespace caffe

#include "caffe/interpolation_layers.hpp"

namespace caffe {

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	interpolation_factor_ =
			this->layer_param_.bilinear_interpolation_param().interpolation_factor();
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->num_axes(), 4);
	vector<int> topShape(4);
	topShape[0] = bottom[0]->shape(0);
	topShape[1] = bottom[0]->shape(1);
	topShape[2] = bottom[0]->shape(2) * interpolation_factor_;
	topShape[3] = bottom[0]->shape(3) * interpolation_factor_;

	top[0]->Reshape(topShape);
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->shape(0);
	const int ch = bottom[0]->shape(1);
	const int in_h = bottom[0]->shape(2);
	const int in_w = bottom[0]->shape(3);
	const int out_h = top[0]->shape(2);
	const int out_w = top[0]->shape(3);

	const Dtype* in_data = bottom[0]->cpu_data();
	Dtype* out_data = top[0]->mutable_cpu_data();

	for (int n = 0; n < num; ++n) {
		for (int h = 0; h < out_h; ++h) {
			Dtype fh = (Dtype) (in_h - 1) * (Dtype) h / (Dtype) (out_h - 1);
			int prev_h = floor(fh);
			int next_h = (prev_h + 1) == in_h ? prev_h : prev_h + 1;
			Dtype dh = fh - prev_h;
			for (int w = 0; w < out_w; ++w) {
				Dtype fw = (Dtype) (in_w - 1) * (Dtype) w / (Dtype) (out_w - 1);
				int prev_w = floor(fw);
				int next_w = (prev_w + 1) == in_w ? prev_w : prev_w + 1;
				Dtype dw = fw - prev_w;

				Dtype pp_weight = (1.0 - dh) * (1.0 - dw);
				Dtype pn_weight = (1.0 - dh) * dw;
				Dtype np_weight = dh * (1.0 - dw);
				Dtype nn_weight = dh * dw;

				for (int c = 0; c < ch; ++c) {
					out_data[top[0]->offset(0, c, h, w)] = pp_weight
							* in_data[bottom[0]->offset(0, c, prev_h, prev_w)]
							+ pn_weight * in_data[bottom[0]->offset(0, c, prev_h, next_w)]
							+ np_weight * in_data[bottom[0]->offset(0, c, next_h, prev_w)]
							+ nn_weight * in_data[bottom[0]->offset(0, c, next_h, next_w)];
				} // for (int c = 0; c < ch; ++c)
			} // for (int w = 0; w < out_w; ++w)
		} // for (int h = 0; h < out_h; ++h)
		in_data += bottom[0]->offset(1);
		out_data += top[0]->offset(1);
	} // for (int n = 0; n < num; ++n)
}

template<typename Dtype>
void BilinearInterpolationLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const int num = bottom[0]->shape(0);
	const int ch = bottom[0]->shape(1);
	const int in_h = bottom[0]->shape(2);
	const int in_w = bottom[0]->shape(3);
	const int out_h = top[0]->shape(2);
	const int out_w = top[0]->shape(3);

	Dtype* in_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* out_diff = top[0]->cpu_diff();

	caffe_memset(bottom[0]->count() * sizeof(Dtype), 0, in_diff);

	for (int n = 0; n < num; ++n) {
		for (int h = 0; h < out_h; ++h) {
			Dtype fh = (Dtype) (in_h - 1) * (Dtype) h / (Dtype) (out_h - 1);
			int prev_h = floor(fh);
			int next_h = (prev_h + 1) == in_h ? prev_h : prev_h + 1;
			Dtype dh = fh - prev_h;
			for (int w = 0; w < out_w; ++w) {
				Dtype fw = (Dtype) (in_w - 1) * (Dtype) w / (Dtype) (out_w - 1);
				int prev_w = floor(fw);
				int next_w = (prev_w + 1) == in_w ? prev_w : prev_w + 1;
				Dtype dw = fw - prev_w;

				Dtype pp_weight = (1.0 - dh) * (1.0 - dw);
				Dtype pn_weight = (1.0 - dh) * dw;
				Dtype np_weight = dh * (1.0 - dw);
				Dtype nn_weight = dh * dw;

				for (int c = 0; c < ch; ++c) {
					const Dtype out_diff_cur = out_diff[top[0]->offset(0, c, h, w)];
					in_diff[bottom[0]->offset(0, c, prev_h, prev_w)] += (pp_weight
							* out_diff_cur);
					in_diff[bottom[0]->offset(0, c, prev_h, next_w)] += (pn_weight
							* out_diff_cur);
					in_diff[bottom[0]->offset(0, c, next_h, prev_w)] += (np_weight
							* out_diff_cur);
					in_diff[bottom[0]->offset(0, c, next_h, next_w)] += (nn_weight
							* out_diff_cur);
				}
			}
		}
		in_diff += bottom[0]->offset(1);
		out_diff += top[0]->offset(1);
	}
}

#ifdef CPU_ONLY
	STUB_GPU(BilinearInterpolationLayer);
#endif

INSTANTIATE_CLASS(BilinearInterpolationLayer);
REGISTER_LAYER_CLASS(BilinearInterpolation);
} // namespace caffe

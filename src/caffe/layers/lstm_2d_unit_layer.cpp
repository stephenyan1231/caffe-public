#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sequence_2d_layers.hpp"

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
	return 2. * sigmoid(2. * x) - 1.;
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom.size(); ++i) {
		// assume bottom shape is (1, n, N)
		CHECK_EQ(bottom[i]->num_axes(), 3);
		CHECK_EQ(bottom[i]->shape(0), 1);
	}
	CHECK_EQ(bottom[0]->shape(2), 25);
	num_ = bottom[0]->shape(1);
	hidden_dim_ = bottom[0]->shape(2);
//	LOG(WARNING)<<"LSTM2DUnitLayer num_ "<<num_<<" hidden_dim_ "<<hidden_dim_;
	CHECK_GT(hidden_dim_, 0);
	CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
	CHECK_EQ(bottom[2]->shape(2), 5 * bottom[1]->shape(2));
	top[0]->ReshapeLike(*bottom[0]);
	top[1]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int good_out_c_c =0,good_out_h_c=0;
	for(int n = 0; n < num_; ++n){
		const Dtype* c_xprev_data = bottom[0]->cpu_data() + n * hidden_dim_;
		const Dtype* c_yprev_data = bottom[1]->cpu_data() + n * hidden_dim_;
		const Dtype* g_data = bottom[2]->cpu_data() + n * hidden_dim_ * 5;
		const Dtype* i_data = g_data + hidden_dim_;
		const Dtype* o_data = i_data + hidden_dim_;
		const Dtype* fx_data = o_data + hidden_dim_;
		const Dtype* fy_data = fx_data + hidden_dim_;
		Dtype* out_c_data = top[0]->mutable_cpu_data() + n * hidden_dim_;
		Dtype* out_h_data = top[1]->mutable_cpu_data() + n * hidden_dim_;
		for (int p = 0; p < hidden_dim_; ++p) {
			const Dtype g = tanh(g_data[p]);
			const Dtype i = sigmoid(i_data[p]);
			const Dtype o = sigmoid(o_data[p]);
			const Dtype fx = sigmoid(fx_data[p]);
			const Dtype fy = sigmoid(fy_data[p]);
			out_c_data[p] = fx * c_xprev_data[p] + fy * c_yprev_data[p] + i * g;
			out_h_data[p] = o * tanh(out_c_data[p]);
			if(out_c_data[p] != 0){
				good_out_c_c++;
			}
			if(out_h_data[p] != 0){
				good_out_h_c++;
			}

		}
	}
	DLOG(WARNING)<<"LSTM2DUnitLayer<Dtype>::Forward_cpu name "
			<<this->layer_param_.name()
			<<" good_out_c_c "<<good_out_c_c<<" good_out_h_c "<<good_out_h_c;
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2]) {
		return;
	}

	for(int n = 0; n < num_; ++n){
		const Dtype* c_xprev_data = bottom[0]->cpu_data() + n * hidden_dim_;
		const Dtype* c_yprev_data = bottom[1]->cpu_data() + n * hidden_dim_;
		const Dtype* g_data = bottom[2]->cpu_data() + n * hidden_dim_ * 5;
		const Dtype* i_data = g_data + hidden_dim_;
		const Dtype* o_data = i_data + hidden_dim_;
		const Dtype* fx_data = o_data + hidden_dim_;
		const Dtype* fy_data = fx_data + hidden_dim_;
		const Dtype* c_data = top[0]->cpu_data() + n * hidden_dim_;
		const Dtype* c_diff = top[0]->cpu_diff() + n * hidden_dim_;
		const Dtype* h_diff = top[1]->cpu_diff() + n * hidden_dim_;
		Dtype* c_xprev_diff = bottom[0]->mutable_cpu_diff() + n * hidden_dim_;
		Dtype* c_yprev_diff = bottom[1]->mutable_cpu_diff() + n * hidden_dim_;
		Dtype* g_diff = bottom[2]->mutable_cpu_diff() + n * hidden_dim_ * 5;
		Dtype* i_diff = g_diff + hidden_dim_;
		Dtype* o_diff = i_diff + hidden_dim_;
		Dtype* fx_diff = o_diff + hidden_dim_;
		Dtype* fy_diff = fx_diff + hidden_dim_;

		for (int p = 0; p < hidden_dim_; ++p) {
				const Dtype g = tanh(g_data[p]);
				const Dtype i = sigmoid(i_data[p]);
				const Dtype o = sigmoid(o_data[p]);
				const Dtype fx = sigmoid(fx_data[p]);
				const Dtype fy = sigmoid(fy_data[p]);
				const Dtype tanh_c = tanh(c_data[p]);
				Dtype factor = c_diff[p] + h_diff[p] * o * (1 - tanh_c * tanh_c);

				if (isnan(c_diff[p]) || isnan(h_diff[p])) {
					LOG(WARNING)<<"p "<<p<<" c_diff[p] "<<c_diff[p]<<
					" h_diff[p] "<<h_diff[p]<<" factor "<<factor
					<<" o "<<o<<" tanh_c "<<tanh_c<<" factor "<<factor;;
				}

				c_xprev_diff[p] = factor * fx;
				c_yprev_diff[p] = factor * fy;
				g_diff[p] = factor * i * (1 - g * g);
				i_diff[p] = factor * g * i * ((Dtype) 1.0 - i);
				o_diff[p] = h_diff[p] * tanh(c_data[p]) * o * ((Dtype) 1.0 - o);
				fx_diff[p] = factor * c_xprev_data[p] * fx * ((Dtype) 1.0 - fx);
				fy_diff[p] = factor * c_yprev_data[p] * fy * ((Dtype) 1.0 - fy);
			}
	}
}

#ifdef CPU_ONLY
	STUB_GPU(LSTM2DUnitLayer);
#endif

INSTANTIATE_CLASS(LSTM2DUnitLayer);
REGISTER_LAYER_CLASS(LSTM2DUnit);

} // namespace caffe

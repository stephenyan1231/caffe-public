#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sequence_2d_layers.hpp"

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
	Dtype ret = 1. / (1. + exp(-x));
	CHECK(!isinf(ret))<<" sigmoid x "<<x;
	CHECK(!isnan(ret))<<" sigmoid x "<<x;
	return ret;
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
	Dtype ret = 2. * sigmoid(2. * x) - 1.;
	CHECK(!isinf(ret))<<" tanh x "<<x;
	CHECK(!isnan(ret))<<" tanh x "<<x;
	return ret;
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom.size(); ++i) {
		// assume bottom shape is (1, n, N)
		CHECK_EQ(bottom[i]->num_axes(), 3);
		CHECK_EQ(bottom[i]->shape(0), 1);
	}
	num_ = bottom[0]->shape(1);
	hidden_dim_ = bottom[0]->shape(2);
	CHECK_GT(hidden_dim_, 0);
	CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
	CHECK_EQ(bottom[2]->shape(2), 5 * bottom[1]->shape(2));
	top[0]->ReshapeLike(*bottom[0]);
	top[1]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	for (int n = 0; n < num_; ++n) {
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
			Dtype i = 0;
			if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_SIGMOID) {
				i = sigmoid(i_data[p]);
			} else if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_TANH) {
				i = tanh(i_data[p]);
			} else {
				LOG(ERROR)<<"Unrecognized input gate activation function ";
			}
			Dtype o = 0;
			if (this->layer_param_.lstm_2d_unit_param().output_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_SIGMOID) {
				o = sigmoid(o_data[p]);
			} else if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_TANH) {
				o = tanh(o_data[p]);
			} else {
				LOG(ERROR)<<"Unrecognized output gate activation function ";
			}
			const Dtype fx = sigmoid(fx_data[p]);
			const Dtype fy = sigmoid(fy_data[p]);
			out_c_data[p] = 0.5 * fx * c_xprev_data[p] + 0.5 * fy * c_yprev_data[p] + i * g;
//			if(p == 1){
//				LOG(WARNING)<<"p "<<p<<" c_xprev_data "<<c_xprev_data[p]<<
//						" c_yprev_data "<<c_yprev_data[p];
//			}
			CHECK(!isinf(out_c_data[p]))<<" layer "<<this->layer_param_.name()<<
					" out_c_data inf p "<<p<<
					" fx "<<fx<<" c_xprev_data "<<c_xprev_data[p]<<
					" fy "<<fy<<" c_yprev_data "<<c_yprev_data[p]<<
					" i "<<i<<" g "<<g;
			out_h_data[p] = o * tanh(out_c_data[p]);
		}
	}

	string layer_name = this->layer_param_.name();
	if(layer_name.find("pp_20_20") != string::npos){
//		const Dtype* out_c_data = top[0]->cpu_data();
//		LOG(WARNING)<<"LSTM2DUnitLayer layer "<<layer_name<<" out_c_data "<<
//				out_c_data[0]<<" "<<out_c_data[1]<<" "<<out_c_data[2];
	}

	const Dtype* top_0_data=top[0]->cpu_data();
	for(int i=0;i<top[0]->count();++i){
		if(isnan(top_0_data[i])){
			LOG(WARNING)<<"i "<<i<<" "<<top_0_data[i];
		}
	}
	const Dtype* top_1_data=top[1]->cpu_data();
	for(int i=0;i<top[1]->count();++i){
		if(isnan(top_1_data[i])){
			LOG(WARNING)<<"i "<<i<<" "<<top_1_data[i];
		}
	}
}

template<typename Dtype>
void LSTM2DUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2]) {
		return;
	}
//	const Dtype* h_diff = top[1]->cpu_diff();
//	LOG(WARNING)<<"LSTM2DUnitLayer<Dtype>::Backward_cpu "
//			<<this->layer_param_.name()<<" top[1] diff";
//	LOG(WARNING)<<h_diff[0]<<" "<<h_diff[1]<<" "<<h_diff[2]<<
//			" "<<h_diff[3];


	for (int n = 0; n < num_; ++n) {
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

			CHECK(!isnan(factor))<<"layer "<<this->layer_param_.name();

			if (isnan(c_diff[p]) || isnan(h_diff[p])) {
				LOG(WARNING)<<"LSTM2DUnitLayer<Dtype>::Backward_cpu layer name "
						<<this->layer_param_.name()
						<<" n "<<n
			<<" p "<<p<<" c_diff[p] "<<c_diff[p]<<
				" h_diff[p] "<<h_diff[p]<<" factor "<<factor
				<<" o "<<o<<" tanh_c "<<tanh_c<<" factor "<<factor;;
			}

			c_xprev_diff[p] = factor * fx * 0.5;
			c_yprev_diff[p] = factor * fy * 0.5;
			g_diff[p] = factor * i * (1 - g * g);
			if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_SIGMOID) {
				i_diff[p] = factor * g * i * ((Dtype) 1.0 - i);
			} else if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_TANH) {
				i_diff[p] = factor * g * ((Dtype) 1.0 - i * i);
			} else {
				LOG(ERROR)<<"Unrecognized input gate activation function ";
			}
			if (this->layer_param_.lstm_2d_unit_param().output_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_SIGMOID) {
				o_diff[p] = h_diff[p] * tanh_c * o * ((Dtype) 1.0 - o);
			} else if (this->layer_param_.lstm_2d_unit_param().input_activation()
					== LSTM2DUnitParameter_ACTIVATION_FUNCTION_TANH) {
				o_diff[p] = h_diff[p] * tanh_c * ((Dtype) 1.0 - o * o);
			} else {
				LOG(ERROR)<<"Unrecognized output gate activation function ";
			}
			fx_diff[p] = factor * c_xprev_data[p] * fx * ((Dtype) 1.0 - fx) * 0.5;
			fy_diff[p] = factor * c_yprev_data[p] * fy * ((Dtype) 1.0 - fy) * 0.5;
			CHECK(!isnan(fx_diff[p]))<<" fx_diff p "<<p<<" factor "<<factor<<" c_xprev_data "<<c_xprev_data[p]<<" fx "<<fx;
			CHECK(!isnan(fy_diff[p]))<<" fy_diff p "<<p<<" factor "<<factor<<" c_yprev_data "<<c_yprev_data[p]<<" fy "<<fy;
		}
	}

	const Dtype* bottom_0_data = bottom[0]->cpu_data();
	for(int i=0;i<bottom[0]->count();++i){
		if(isnan(bottom_0_data[i])){
			LOG(WARNING)<<"i "<<i<<" "<<bottom_0_data[i];
		}
	}

	const Dtype* bottom_1_data = bottom[1]->cpu_data();
	for(int i=0;i<bottom[1]->count();++i){
		if(isnan(bottom_1_data[i])){
			LOG(WARNING)<<"i "<<i<<" "<<bottom_1_data[i];
		}
	}

	const Dtype* bottom_3_diff = bottom[2]->cpu_diff();
	for(int i=0;i<bottom[2]->count();++i){
		if(isnan(bottom_3_diff[i])){
			LOG(WARNING)<<"i "<<i<<" "<<bottom_3_diff[i];
		}
	}

}

#ifdef CPU_ONLY
STUB_GPU(LSTM2DUnitLayer);
#endif

INSTANTIATE_CLASS(LSTM2DUnitLayer);
REGISTER_LAYER_CLASS(LSTM2DUnit);

} // namespace caffe

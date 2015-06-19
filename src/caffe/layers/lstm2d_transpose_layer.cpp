#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void LSTM2DTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// bottom shape should be (1, num, dim)
	CHECK_EQ(bottom[0]->num_axes(), 3);
	CHECK_EQ(bottom[0]->shape(0), 1);

	vector<int> topShape(3);
	topShape[0] = bottom[0]->shape(1);
	topShape[1] = bottom[0]->shape(2);
	topShape[2] = 1;
	top[0]->Reshape(topShape);
}

template<typename Dtype>
void LSTM2DTransposeLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top[0]->ShareData(*bottom[0]);
}

template<typename Dtype>
void LSTM2DTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// TO DO, back-propagate should depend on "propagate_down"
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();
	caffe_copy(top[0]->count(), top_diff, bottom_diff);

	DLOG(WARNING)<<"LSTM2DTransposeLayer<Dtype>::Backward_cpu layer name "
	<<this->layer_param_.name()<<" top[0] asum_diff "<<top[0]->asum_diff()
	<<" bottom[0] asum_diff "<<bottom[0]->asum_diff();
}

#ifdef CPU_ONLY
	STUB_GPU(LSTM2DTransposeLayer);
#endif

INSTANTIATE_CLASS(LSTM2DTransposeLayer);
REGISTER_LAYER_CLASS(LSTM2DTranspose);

} // namespace caffe

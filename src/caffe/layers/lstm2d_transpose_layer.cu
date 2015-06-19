#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void LSTM2DTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->ShareData(*bottom[0]);
}


template<typename Dtype>
void LSTM2DTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();
	caffe_copy(top[0]->count(), top_diff, bottom_diff);}

INSTANTIATE_LAYER_GPU_FUNCS(LSTM2DTransposeLayer);

} // namespace caffe

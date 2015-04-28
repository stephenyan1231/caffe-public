#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	if (Caffe::phase() == Caffe::TEST) {
		this->AssembleParameterMatrix();
	}

	const Dtype* weight = this->blobs_[0]->gpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
//		LOG(INFO)<<"conv layer name "<<this->layer_param_.name()
//				<<" bottom shape "<<bottom[i]->channels()<<" "
//				<<bottom[i]->height()<<" "<<bottom[i]->width();
//		LOG(INFO)<<"conv layer name "<<this->layer_param_.name()
//				<<" top shape "<<top[i]->channels()<<" "
//				<<top[i]->height()<<" "<<top[i]->width();

		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
					top_data + top[i]->offset(n));
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
			}
		}
	}
//	LOG(INFO)<<"ConvolutionLayer<Dtype>::Forward_gpu "<<this->layer_param_.name()
//			<<" top shape "<<
//			top[0]->num()<<" "<<top[0]->channels()<<" "<<
//			top[0]->height()<<" "<<top[0]->width()<<" "<<
//			top[0]->count()*sizeof(Dtype);
	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
//		LOG(INFO)<<"conv layer name "<<this->layer_param_.name();
//		size_t free_mem, total_mem;
//		cudaMemGetInfo(&free_mem, &total_mem);
//		LOG(INFO)<<"before: free memoey "<<free_mem<<" total_mem "<<total_mem;


		for (int i = 0; i < bottom.size(); ++i) {
//			LOG(INFO)<<"conv layer name "<<this->layer_param_.name()
//					<<" release bottom blob count "<<bottom[i]->count();
			bottom[i]->ReshapeForceMemoryFree(0, 0, 0, 0);
		}
//		DLOG(INFO)<<"conv layer name "<<this->layer_param_.name()<<
//				" force_free_col_buffer_bias_multiplier_gpu_memory";
		this->force_free_col_buffer_bias_multiplier_gpu_memory();
//		DLOG(INFO)<<"conv layer name "<<this->layer_param_.name()<<
//				" force_free_col_buffer_bias_multiplier_gpu_memory end";
//		LOG(INFO)<<this->layer_param_.name()<<" ConvolutionLayer<Dtype>::Forward_gpu free memory";

//		cudaMemGetInfo(&free_mem, &total_mem);
//		LOG(INFO)<<"after: free memoey "<<free_mem<<" total_mem "<<total_mem;

	}

	if (Caffe::phase() == Caffe::TEST) {
		this->FreeParameterMatrix();
	}
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	if (this->param_propagate_down_[0]) {
		caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
	}
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
				this->blobs_[1]->mutable_gpu_diff());
	}
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->gpu_diff();
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
			}
		}
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
							top_diff + top[i]->offset(n), weight_diff);
				}
				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
					this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
							bottom_diff + bottom[i]->offset(n));
				}
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe

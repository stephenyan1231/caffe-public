#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
		this->AssembleParameterMatrix();
	}
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype) 1.,
			bottom_data, weight, (Dtype) 0., top_data);
	if (bias_term_) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
				bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), (Dtype) 1.,
				top_data);
	}

	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
		this->FreeParameterMatrix();
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		// Gradient with respect to weight
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype) 1.,
				top_diff, bottom_data, (Dtype) 0., this->blobs_[0]->mutable_gpu_diff());
	}
	if (bias_term_ && this->param_propagate_down_[1]) {
		const Dtype* top_diff = top[0]->gpu_diff();
		// Gradient with respect to bias
		caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
				bias_multiplier_.gpu_data(), (Dtype) 0.,
				this->blobs_[1]->mutable_gpu_diff());
	}
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->gpu_diff();
		// Gradient with respect to bottom data
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype) 1.,
				top_diff, this->blobs_[0]->gpu_data(), (Dtype) 0.,
				bottom[0]->mutable_gpu_diff());
	}
}

template<typename Dtype>
__global__ void AssembleMatrix(const int nthreads, const Dtype* centers_data,
		const Dtype* indices_data, int num_center, int num_seg, int mat_height,
		int mat_width, Dtype* mat_data) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int seg_size = mat_width / num_seg;
		int mat_y = index / num_seg;
		int mat_x = index % num_seg;
		int index = static_cast<int>(indices_data[mat_y * num_seg + mat_x]);
		centers_data += (index * mat_width + mat_x * seg_size);
		mat_data += (mat_y * mat_width + mat_x * seg_size);
		for (int i = 0; i < seg_size; ++i) {
			mat_data[i] = centers_data[i];
		}
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::AssembleParameterMatrix() {
	if (quantization_kmean_num_cluster_ > 0) {
//	if (quantization_kmean_num_cluster_ > 0 && !parameter_matrix_assembled_) {
//		LOG(INFO)<<"InnerProductLayer<Dtype>::AssembleParameterMatrix";
		Dtype* param_data = NULL;
//		const Dtype* centers_data = quantization_kmean_cluster_centers_.cpu_data();
		const Dtype* indices_data = quantization_kmean_cluster_indices_.cpu_data();
		this->blobs_[0]->Reshape(1, 1, N_, K_);
		CHECK_EQ(K_ % quantization_num_segment_, 0);
//		const int segment_size = K_ / quantization_num_segment_;

		switch (Caffe::mode()) {
			case Caffe::CPU:
			NOT_IMPLEMENTED;
			break;

			case Caffe::GPU:
			AssembleMatrix<Dtype><<<CAFFE_GET_BLOCKS(N_*quantization_num_segment_),CAFFE_CUDA_NUM_THREADS>>>
					(N_*quantization_num_segment_,quantization_kmean_cluster_centers_.gpu_data(),
							quantization_kmean_cluster_indices_.gpu_data(), quantization_kmean_cluster_centers_.height(),
							quantization_num_segment_, N_, K_, this->blobs_[0]->mutable_gpu_data());

//			param_data = this->blobs_[0]->mutable_gpu_data();
//			for (int i = 0; i < N_; ++i) {
//				for (int j = 0; j < quantization_num_segment_; ++j) {
//					int index = static_cast<int>((indices_data
//									+ quantization_kmean_cluster_indices_.offset(0,0,i))[j]);
//					caffe_copy(segment_size,
//							centers_data + quantization_kmean_cluster_centers_.offset(0,0,index)
//							+ j * segment_size,
//							param_data + this->blobs_[0]->offset(0,0,i) + j * segment_size);
//				}
//			}
			break;
			default:
			LOG(FATAL)<< "Unknown caffe mode: " << Caffe::mode();
		}
//		parameter_matrix_assembled_ = true;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe

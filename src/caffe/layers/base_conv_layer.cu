#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/matrix_quantization.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::FreeParameterMatrix() {
	if (this->parameter_compress_ && quantization_kmean_num_cluster_ > 0) {
		this->blobs_[0]->ReshapeForceMemoryFree(0, 0, 0, 0);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::AssembleParameterMatrix() {
	if (this->parameter_compress_ && quantization_kmean_num_cluster_ > 0) {
		LOG(INFO) << "BaseConvolutionLayer<Dtype>::AssembleParameterMatrix";
//		const Dtype* indices_data = quantization_kmean_cluster_indices_.cpu_data();
		const int mat_height = conv_out_channels_;
		const int mat_width = conv_in_channels_ * kernel_h_ * kernel_w_ / group_;
		this->blobs_[0]->Reshape(conv_out_channels_, conv_in_channels_ / group_,
				kernel_h_, kernel_w_);
		CHECK_EQ(mat_width % quantization_num_segment_, 0);

		switch (Caffe::mode()) {
		case Caffe::CPU:
			NOT_IMPLEMENTED;
			break;

		case Caffe::GPU:
			AssembleMatrix<Dtype> <<<
					CAFFE_GET_BLOCKS(mat_height * quantization_num_segment_),
					CAFFE_CUDA_NUM_THREADS>>>(mat_height * quantization_num_segment_,
					quantization_kmean_cluster_centers_.gpu_data(),
//					quantization_kmean_cluster_indices_uint16_.gpu_data(),
					quantization_kmean_cluster_indices_uint8_.gpu_data(),
					quantization_kmean_cluster_centers_.height(),
					quantization_num_segment_, mat_height, mat_width,
					this->blobs_[0]->mutable_gpu_data());
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}
}

template void BaseConvolutionLayer<float>::FreeParameterMatrix();
template void BaseConvolutionLayer<double>::FreeParameterMatrix();
template void BaseConvolutionLayer<float>::AssembleParameterMatrix();
template void BaseConvolutionLayer<double>::AssembleParameterMatrix();

} // namespace caffe

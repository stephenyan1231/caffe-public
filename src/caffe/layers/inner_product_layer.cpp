#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  K_ = bottom[0]->count() / bottom[0]->num();

  quantization_kmean_num_cluster_ = this->layer_param_.inner_product_param().quantization_kmean_num_cluster();
  quantization_num_segment_ = this->layer_param_.inner_product_param().quantization_num_segment();
  quantization_kmean_cluster_centers_file_ = this->layer_param_.inner_product_param().quantization_kmean_cluster_centers_file();
  quantization_kmean_cluster_indices_file_ = this->layer_param_.inner_product_param().quantization_kmean_cluster_indices_file();
  parameter_matrix_assembled_ = false;
  if(quantization_kmean_num_cluster_ >0){
  	CHECK_GE(quantization_num_segment_, 1);
  	LOG(INFO)<<"Layer "<<this->layer_param_.name()<<" read quantization_kmean_cluster_centers_ from "
  			<<quantization_kmean_cluster_centers_file_;
  	BlobProto centers_blob_proto;
  	ReadProtoFromBinaryFile(quantization_kmean_cluster_centers_file_, &centers_blob_proto);
  	quantization_kmean_cluster_centers_.FromProto(centers_blob_proto);
  	LOG(INFO)<<"quantization_kmean_cluster_centers_ height "<<quantization_kmean_cluster_centers_.height()
  			<<" width "<<quantization_kmean_cluster_centers_.width();

  	BlobProto indices_blob_proto;
  	ReadProtoFromBinaryFile(quantization_kmean_cluster_indices_file_, &indices_blob_proto);
  	quantization_kmean_cluster_indices_.FromProto(indices_blob_proto);
  	LOG(INFO)<<"quantization_kmean_cluster_indices_ height "<<quantization_kmean_cluster_indices_.height()
  			<<" width "<<quantization_kmean_cluster_indices_.width();
  }

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    if(quantization_kmean_num_cluster_ == 0){
      // Intialize the weight
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }else{
      this->blobs_[0].reset(new Blob<Dtype>(0, 0, 0, 0));
    }

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";
  top[0]->Reshape(bottom[0]->num(), N_, 1, 1);
  // Set up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}


template<typename Dtype>
void InnerProductLayer<Dtype>::FreeParameterMatrix() {
	if(quantization_kmean_num_cluster_ > 0 ){
//		LOG(INFO)<<"InnerProductLayer<Dtype>::FreeParameterMatrix";
		this->blobs_[0]->ReshapeForceMemoryFree(0, 0, 0, 0);
	}
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe

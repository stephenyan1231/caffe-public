// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CCCPLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1)
      << "CCCP Pooling Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1)
      << "CCCP Pooling Layer takes a single blob as output.";

  num_output_ = this->layer_param_.cccp_param().num_output();
  group_      = this->layer_param_.cccp_param().group();
  bias_term_   = this->layer_param_.cccp_param().bias_term();

  // Figure out the dimensions
  channels_ = bottom[0]->channels();
  width_ = bottom[0]->width();
  height_ = bottom[0]->height();
  num_ = bottom[0]->num();

  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);

  (*top)[0]->Reshape(bottom[0]->num(), group_ * num_output_, height_, width_);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, group_ * num_output_,
            channels_ / group_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.cccp_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, group_ * num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.cccp_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(width_ * height_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < width_ * height_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}

template <typename Dtype>
Dtype CCCPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int weight_offset = num_output_ * channels_ / group_;
  const int bottom_group_offset = width_ * height_ * channels_ / group_;
  const int top_group_offset = width_ * height_ * num_output_;

  for (int n = 0; n < num_; ++n) {
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          width_ * height_, channels_ / group_, (Dtype)1.,
          weight + g * weight_offset,
          bottom_data + bottom[0]->offset(n) + g * bottom_group_offset,
          (Dtype)0., top_data + (*top)[0]->offset(n) + g * top_group_offset);
    }
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, group_ * num_output_,
          width_ * height_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void CCCPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = NULL;
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  const int weight_offset = num_output_ * channels_ / group_;
  const int bottom_group_offset = width_ * height_ * channels_ / group_;
  const int top_group_offset = width_ * height_ * num_output_;

  // Gradient w.r.t. bias
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_ * group_,
          width_ * height_, (Dtype)1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., bias_diff);
    }
  }

  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
    // The gradient will be accumulated
    for (int g = 0; g < group_; ++g) {
      // Gradient with respect to weight
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_output_,
          channels_ / group_, width_ * height_, (Dtype)1.,
          top_diff + top[0]->offset(n) + g * top_group_offset,
          bottom_data + (*bottom)[0]->offset(n) + g * bottom_group_offset,
          (Dtype)1., weight_diff + g * weight_offset);
      // Gradient w.r.t. bottom data if necessary
      if (propagate_down) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_ / group_,
            width_ * height_, num_output_, (Dtype)1.,
            weight + g * weight_offset,
            top_diff + top[0]->offset(n) + g * top_group_offset, (Dtype)0.,
            bottom_diff + (*bottom)[0]->offset(n) + g * bottom_group_offset);
      }
    }
  }
}

INSTANTIATE_CLASS(CCCPLayer);

}  // namespace caffe

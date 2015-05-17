// Copyright 2015 Zhicheng Yan

#include "caffe/layer.hpp"
#include "caffe/video_enhancement_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ImageEnhancementSquareDiffLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Layer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "The data and label should have the same number.";
	for (int i = 0; i < 3; ++i) {
		CHECK_EQ(bottom[i]->height(), 1);
		CHECK_EQ(bottom[i]->width(), 1);
	}
	// LossLayers have a non-zero (1) loss by default.
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
	}
}

template<typename Dtype>
void ImageEnhancementSquareDiffLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(1, 1, 1, 1);
	CHECK_EQ(bottom[1]->channels() % color_basis_dim_, 0);
	pixel_samples_num_per_segment_ = bottom[1]->channels() / color_basis_dim_;
	pred_LAB_color_.Reshape(bottom[0]->num(),
			pixel_samples_num_per_segment_ * color_dim_, 1, 1);
}

template<typename Dtype>
void ImageEnhancementSquareDiffLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* regressed_coef = bottom[0]->cpu_data();
	const Dtype* quad_color_basis = bottom[1]->cpu_data();
	const Dtype* gt_LAB_color = bottom[2]->cpu_data();
	// pred_LAB_color (num, n*3, 1, 1)
	Dtype* pred_LAB_color = pred_LAB_color_.mutable_cpu_data();
	CHECK_EQ(pred_LAB_color_.num(), bottom[2]->num());
	CHECK_EQ(pred_LAB_color_.channels(), bottom[2]->channels());

	int num = bottom[0]->num();
	for (int i = 0; i < num; ++i) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				pixel_samples_num_per_segment_, color_dim_, color_basis_dim_,
				(Dtype) 1., quad_color_basis + bottom[1]->offset(i),
				regressed_coef + bottom[0]->offset(i), (Dtype) 0., pred_LAB_color + pred_LAB_color_.offset(i));
	}
	// Now pred_LAB_color stores the difference between predicted color and ground truth color
	for (int i = 0; i < pred_LAB_color_.count(); ++i) {
		pred_LAB_color[i] -= gt_LAB_color[i];
	}
	Dtype square_diff = pred_LAB_color_.sumsq_data()
			/ (2 * num * pixel_samples_num_per_segment_);
	Dtype *top_data = top[0]->mutable_cpu_data();
	top_data[0] = square_diff;
}

template<typename Dtype>
void ImageEnhancementSquareDiffLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype *color_diff = pred_LAB_color_.cpu_data();
	const Dtype *quad_color_basis = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	caffe_memset(sizeof(Dtype) * bottom[0]->count(), 0, bottom_diff);

	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < color_dim_; ++j) {
			int bottom_diff_offset = bottom[0]->offset(i, j * color_basis_dim_);
			for (int k = 0; k < pixel_samples_num_per_segment_; ++k) {
				int color_diff_offset = pred_LAB_color_.offset(i, k * color_dim_ + j);
				int quad_color_basis_offset = bottom[1]->offset(i,
						k * color_basis_dim_);
				for (int p = 0; p < color_basis_dim_; ++p) {
					bottom_diff[bottom_diff_offset + p] += ((Dtype) 1.
							/ (num * pixel_samples_num_per_segment_))
							* color_diff[color_diff_offset]
							* quad_color_basis[quad_color_basis_offset + p];
				}
			}
		}
	}
}

INSTANTIATE_CLASS(ImageEnhancementSquareDiffLossLayer);
REGISTER_LAYER_CLASS(ImageEnhancementSquareDiffLoss);

} // namespace caffe

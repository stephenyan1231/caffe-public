// Copyright 2015 Zhicheng Yan

#include "caffe/shift_stitch_layer.hpp"

namespace caffe {

template<typename Dtype>
void ShiftStitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	ShiftStitchParameter pool_param = this->layer_param_.shift_stitch_param();

	iter_ = pool_param.iter();
	CHECK(
			(pool_param.stride_size() == 0)
			!= !((pool_param.stride_h_size() > 0)
					&& (pool_param.stride_w_size() > 0)))
																										<< "Stride size is stride OR stride_h and stride_w: not both ";
	CHECK((pool_param.stride_size() > 0)||((pool_param.stride_h_size() > 0) && (pool_param.stride_w_size() > 0)))
																																																										<< "For non-uniform stride, both stride_h and stride_w are required";

	stride_h_.resize(iter_);
	stride_w_.resize(iter_);
	if (pool_param.stride_h_size() == 0) {
		CHECK_EQ(iter_, pool_param.stride_size());
		for (int i = 0; i < iter_; ++i) {
			stride_h_[i] = pool_param.stride(i);
			stride_w_[i] = pool_param.stride(i);
			DLOG(INFO)<<"ShiftStitchLayer<Dtype>::LayerSetUp i "<<i<<" stride_h_ "
					<<stride_h_[i]<<" stride_w_ "<<stride_w_[i];
		}
	} else {
		CHECK_EQ(iter_, pool_param.stride_h_size());
		CHECK_EQ(iter_, pool_param.stride_w_size());
		for (int i = 0; i < iter_; ++i) {
			stride_h_[i] = pool_param.stride_h(i);
			stride_w_[i] = pool_param.stride_w(i);
		}
	}
}

template<typename Dtype>
void ShiftStitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	out_num_ = num_;
	out_height_ = height_;
	out_width_ = width_;
	for (int i = 0; i < iter_; ++i) {
		out_num_ /= (stride_h_[i] * stride_w_[i]);
		out_height_ *= stride_h_[i];
		out_width_ *= stride_w_[i];
	}
	top[0]->Reshape(out_num_, channels_, out_height_, out_width_);
}

template<typename Dtype>
void ShiftStitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int iter_out_num = num_;
	int iter_out_height = height_;
	int iter_out_width = width_;
	Blob<Dtype> *src_blob = NULL, *tgt_blob = NULL;

	for (int i = 0; i < iter_; ++i) {
		iter_out_num /= (stride_h_[i] * stride_w_[i]);
		iter_out_height *= stride_h_[i];
		iter_out_width *= stride_w_[i];
		if (i == 0) {
			src_blob = bottom[0];
		} else {
			src_blob = tgt_blob;
		}
		if (i == (iter_ - 1)) {
			tgt_blob = top[0];
		} else {
			tgt_blob = new Blob<Dtype>(iter_out_num, channels_, iter_out_height,
					iter_out_width);
		}
		const Dtype* src_data = src_blob->cpu_data();
		Dtype* tgt_data;
		for (int src_ptr = 0, n = 0; n < iter_out_num; ++n) {
			for (int sy = 0; sy < stride_h_[i]; ++sy) {
				for (int sx = 0; sx < stride_w_[i]; ++sx) {
					for (int ch = 0; ch < channels_; ++ch) {
						tgt_data = tgt_blob->mutable_cpu_data() + tgt_blob->offset(n, ch);
						for (int h = 0; h < src_blob->height(); ++h) {
							int th = h * stride_h_[i] + sy;
							for (int w = 0; w < src_blob->width(); ++w, ++src_ptr) {
								int tw = w * stride_w_[i] + sx;
								tgt_data[th * iter_out_height + tw] = src_data[src_ptr];
							}
						}
					}
				}
			}
		}
		if (i > 0) {
			delete src_blob;
		}
	}
}

template<typename Dtype>
void ShiftStitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int iter_in_num = out_num_;
	int iter_in_height = out_height_;
	int iter_in_width = out_width_;
	Blob<Dtype> *src_blob, *tgt_blob;
	for (int i = iter_ - 1; i >= 0; --i) {
		iter_in_num *= (stride_h_[i] * stride_w_[i]);
		iter_in_height /= stride_h_[i];
		iter_in_width /= stride_w_[i];
		if (i == (iter_ - 1)) {
			tgt_blob = top[0];
		} else {
			tgt_blob = src_blob;
		}
		if (i == 0) {
			src_blob = bottom[0];
		} else {
			src_blob = new Blob<Dtype>(iter_in_num, channels_, iter_in_height,
					iter_in_width);
		}
		const Dtype* tgt_diff;
		Dtype* src_diff = src_blob->mutable_cpu_diff();
		for (int src_ptr = 0, n = 0; n < tgt_blob->num(); ++n) {
			for (int sy = 0; sy < stride_h_[i]; ++sy) {
				for (int sx = 0; sx < stride_w_[i]; ++sx) {
					for (int ch = 0; ch < channels_; ++ch) {
						tgt_diff = tgt_blob->cpu_diff() + tgt_blob->offset(n, ch);
						for (int h = 0; h < src_blob->height(); ++h) {
							int th = h * stride_h_[i] + sy;
							for (int w = 0; w < src_blob->width(); ++w, ++src_ptr) {
								int tw = w * stride_w_[i] + sx;
								src_diff[src_ptr] = tgt_diff[th * tgt_blob->height() + tw];
							}
						}
					}
				}
			}
		}
		if (i < (iter_ - 1)) {
			delete tgt_blob;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ShiftStitchLayer);
#endif

INSTANTIATE_CLASS(ShiftStitchLayer);
REGISTER_LAYER_CLASS(ShiftStitch);
} // namespace caffe

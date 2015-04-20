//Copyright 2015 Zhicheng Yan

#include "caffe/hdcnn_layers.hpp"

namespace caffe {

template<typename Dtype>
void CompactProbabilisticAverageProbLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	prob_num_ = bottom.size() - 1;
	CHECK_EQ(top.size(), 1);

	class_num_ =
			this->layer_param_.compact_probabilistic_average_prob_layer_param().num_class();
	compact_layer_size_.resize(prob_num_);
	compact_layer_class_id_.resize(prob_num_);

	CHECK_EQ(prob_num_,
			this->layer_param_.compact_probabilistic_average_prob_layer_param().compact_classify_layer_param_size());

	for (int i = 0; i < prob_num_; ++i) {
		compact_layer_size_[i] =
				this->layer_param_.compact_probabilistic_average_prob_layer_param().compact_classify_layer_param(
						i).class_id_size();
		compact_layer_class_id_[i].resize(compact_layer_size_[i]);
//		CHECK_EQ(bottom[i]->count()/bottom[i]->num(),compact_layer_size_[i]);
		for (int j = 0; j < compact_layer_size_[i]; ++j) {
			compact_layer_class_id_[i][j] =
					this->layer_param_.compact_probabilistic_average_prob_layer_param().compact_classify_layer_param(
							i).class_id(j);
		}
	}
}

template<typename Dtype>
void CompactProbabilisticAverageProbLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	spatial_h_ = bottom[0]->height();
	spatial_w_ = bottom[0]->width();
	DLOG(INFO)
			<< "CompactProbabilisticAverageProbLayer<Dtype>::Reshape spatial size "
			<< spatial_h_ << " " << spatial_w_;

	for (int i = 0; i < prob_num_; ++i) {
		CHECK_EQ(bottom[i]->num(), bottom[0]->num());
		CHECK_EQ(bottom[i]->channels(), compact_layer_size_[i]);
		CHECK_EQ(bottom[i]->height(), spatial_h_);
		CHECK_EQ(bottom[i]->width(), spatial_w_);
	}

	CHECK_EQ(bottom[prob_num_]->num(), bottom[0]->num());
	CHECK_EQ(bottom[prob_num_]->channels(), prob_num_);
	CHECK_EQ(bottom[prob_num_]->height(), spatial_h_);
	CHECK_EQ(bottom[prob_num_]->width(), spatial_w_);

	top[0]->Reshape(bottom[prob_num_]->num(), class_num_, spatial_h_, spatial_w_);
}

template<typename Dtype>
void CompactProbabilisticAverageProbLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const Dtype* branch_prob_data = bottom[prob_num_]->cpu_data();
	Dtype* top_data = (top)[0]->mutable_cpu_data();
	memset(top_data, 0, sizeof(Dtype) * (top)[0]->count());
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < prob_num_; ++j) {
			const Dtype* prob_data = bottom[j]->cpu_data();
			for (int k = 0; k < compact_layer_size_[j]; ++k) {
				const int class_id = compact_layer_class_id_[j][k];
				for (int y = 0; y < spatial_h_; ++y) {
					for (int x = 0; x < spatial_w_; ++x) {
//						Dtype prob_weight = (branch_prob_data + bottom[prob_num_]->offset(i))[j];
						Dtype prob_weight = branch_prob_data[bottom[prob_num_]->offset(i, j,
								y, x)];
						(top_data + top[0]->offset(i, class_id, y, x))[0] += prob_weight
								* (prob_data + bottom[j]->offset(i, k, y, x))[0];
					}
				}
//				(top_data + (top)[0]->offset(i))[class_id] += prob_weight
//						* (prob_data + bottom[j]->offset(i))[k];
			}
		}
	}
}

template<typename Dtype>
void CompactProbabilisticAverageProbLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const int num = (bottom)[0]->num();
	const Dtype* branch_prob_data = (bottom)[prob_num_]->cpu_data();
	Dtype* branch_prob_diff = (bottom)[prob_num_]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();

	if (propagate_down[0]) {
		memset(branch_prob_diff, 0, sizeof(Dtype) * (bottom)[prob_num_]->count());
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < prob_num_; ++j) {
				const Dtype* bottom_data = bottom[j]->cpu_data();
				Dtype* bottom_diff = bottom[j]->mutable_cpu_diff()
						+ bottom[j]->offset(i);
				const Dtype* top_diff_i = top_diff + top[0]->offset(i);
				Dtype prob_weight =
						(branch_prob_data + bottom[prob_num_]->offset(i))[j];
				for (int k = 0; k < compact_layer_size_[j]; ++k) {
					int class_id = compact_layer_class_id_[j][k];
					bottom_diff[k] = top_diff_i[class_id] * prob_weight;
					(branch_prob_diff + bottom[prob_num_]->offset(i))[j] +=
							top_diff_i[class_id] * (bottom_data + bottom[j]->offset(i))[k];
				}
			}
		}
	}
}

INSTANTIATE_CLASS(CompactProbabilisticAverageProbLayer);
REGISTER_LAYER_CLASS(CompactProbabilisticAverageProb);
} // namespace caffe

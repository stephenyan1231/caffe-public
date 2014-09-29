//Copyright 2014 Zhicheng Yan@eBay

#include "caffe/ensemble_prob_layers.hpp"

namespace caffe {

template<typename Dtype>
ProbabilisticAverageProbLayer<Dtype>::~ProbabilisticAverageProbLayer() {
	free((void*) bottom_prob_ptrs_);
	free((void*) bottom_prob_diff_ptrs_);
	CUDA_CHECK(cudaFree((void* )bottom_prob_ptrs_dev_));
	CUDA_CHECK(cudaFree((void* )bottom_prob_diff_ptrs_dev_));

}

template<typename Dtype>
void ProbabilisticAverageProbLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	prob_num_ = bottom.size() - 1;
	class_num_ = bottom[0]->count() / bottom[0]->num();


	CHECK_EQ(prob_num_, bottom[prob_num_]->count() / bottom[prob_num_]->num());

	CHECK_EQ(top->size(), 1)
			<< "ProbabilisticverageProbLayer should have a single blob as output";

	for (int i = 1; i < prob_num_; ++i) {
		CHECK_EQ(bottom[i - 1]->num(), bottom[i]->num());
		CHECK_EQ(bottom[i - 1]->channels(), bottom[i]->channels());
		CHECK_EQ(bottom[i - 1]->height(), bottom[i]->height());
		CHECK_EQ(bottom[i - 1]->width(), bottom[i]->width());
	}
	(*top)[0]->ReshapeLike(*bottom[0]);

	bottom_prob_ptrs_ = (Dtype**) malloc(sizeof(Dtype*) * prob_num_);
	for (int i = 0; i < prob_num_; ++i)
		bottom_prob_ptrs_[i] = bottom[i]->mutable_gpu_data();
	CUDA_CHECK(
			cudaMalloc((void** )&bottom_prob_ptrs_dev_, sizeof(Dtype*) * prob_num_));
	CUDA_CHECK(
			cudaMemcpy((void* ) bottom_prob_ptrs_dev_,
					(const void* ) bottom_prob_ptrs_, sizeof(Dtype*) * prob_num_,
					cudaMemcpyHostToDevice));
	bottom_prob_diff_ptrs_ = (Dtype**) malloc(sizeof(Dtype*) * prob_num_);
	for (int i = 0; i < prob_num_; ++i)
		bottom_prob_diff_ptrs_[i] = bottom[i]->mutable_gpu_diff();
	CUDA_CHECK(
			cudaMalloc((void** )&bottom_prob_diff_ptrs_dev_,
					sizeof(Dtype*) * prob_num_));
	CUDA_CHECK(
			cudaMemcpy((void* ) bottom_prob_diff_ptrs_dev_,
					(const void* )bottom_prob_diff_ptrs_, sizeof(Dtype*) * prob_num_,
					cudaMemcpyHostToDevice));
}

template<typename Dtype>
Dtype ProbabilisticAverageProbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int num = bottom[0]->num();
	const Dtype* branch_prob_data = bottom[prob_num_]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	memset(top_data, 0, sizeof(Dtype) * (*top)[0]->count());
	for (int i = 0; i < num; ++i) {
		float weight_sum = 0;
		for (int j = 0; j < prob_num_; ++j) {
			const Dtype* prob_data = bottom[j]->cpu_data();
			Dtype prob_weight = (branch_prob_data + bottom[prob_num_]->offset(i))[j];
			for (int k = 0; k < class_num_; ++k) {
				(top_data + (*top)[0]->offset(i))[k] += prob_weight
						* (prob_data + bottom[j]->offset(i))[k];
			}
		}
	}
	return Dtype(0.);
}

template<typename Dtype>
void ProbabilisticAverageProbLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	const int num = (*bottom)[0]->num();
	const Dtype* branch_prob_data = (*bottom)[prob_num_]->cpu_data();
	Dtype* branch_prob_diff = (*bottom)[prob_num_]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();

	if (propagate_down) {
		memset(branch_prob_diff, 0, sizeof(Dtype) * (*bottom)[prob_num_]->count());
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < prob_num_; ++j) {
				const Dtype* bottom_data = (*bottom)[j]->cpu_data();
				Dtype* bottom_diff = (*bottom)[j]->mutable_cpu_diff();
				Dtype prob_weight = (branch_prob_data + (*bottom)[prob_num_]->offset(i))[j];
				for (int k = 0; k < class_num_; ++k) {
					(bottom_diff + (*bottom)[j]->offset(i))[k] = (top_diff
							+ top[0]->offset(i))[k] * prob_weight;
					(branch_prob_diff + (*bottom)[prob_num_]->offset(i))[j] +=
							(top_diff + top[0]->offset(i))[k] * (bottom_data + (*bottom)[j]->offset(i))[k];
				}
			}
		}
	}

}

INSTANTIATE_CLASS(ProbabilisticAverageProbLayer);


} // namespace caffe

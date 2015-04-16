#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/spatial_prob_aggregation_layer.hpp"

namespace caffe {

template<typename Dtype>
void SpatialProbAggregationLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top_k_ = this->layer_param_.spatial_prob_aggregation_param().top_k();
	spatial_subsample_height_factor_ =
			this->layer_param_.spatial_prob_aggregation_param().spatial_subsample_height_factor();
	spatial_subsample_width_factor_ =
			this->layer_param_.spatial_prob_aggregation_param().spatial_subsample_width_factor();
}

template<typename Dtype>
void SpatialProbAggregationLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_LE(top_k_, bottom[0]->channels())
			<< "top_k must be less than or equal to the number of classes.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);

	CHECK_EQ(bottom[2]->num(), bottom[1]->num());
	CHECK_EQ(bottom[2]->channels(), 1);
	CHECK_EQ(bottom[2]->height(), 1);
	CHECK_EQ(bottom[2]->width(), 2);

	top[0]->Reshape(1, 1, 1, 1);
}

template<typename Dtype>
void SpatialProbAggregationLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype accuracy = 0;
	const Dtype *prob_data = bottom[0]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->channels();
	int h = bottom[0]->height();
	int w = bottom[0]->width();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const Dtype* image_size_data = bottom[2]->cpu_data();
	int eh = h, ew = w;
	vector<Dtype> maxval(top_k_ + 1);
	vector<int> max_id(top_k_ + 1);

	for (int i = 0; i < num; ++i) {
		eh = (image_size_data + bottom[2]->offset(i))[0]
				/ spatial_subsample_height_factor_;
		ew = (image_size_data + bottom[2]->offset(i))[1]
				/ spatial_subsample_width_factor_;
		LOG(INFO)<<"i "<<i<<" eh "<<eh<<" ew "<<ew;
		std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
		for (int j = 0; j < dim; ++j) {
			bottom_data_vector[j].first = 0;
			bottom_data_vector[j].second = j;
		}
		for (int y = 0; y < eh; ++y) {
			for (int x = 0; x < ew; ++x) {
				for (int j = 0; j < dim; ++j) {
					bottom_data_vector[j].first += prob_data[(j * h + y) * w + x];
				}
			}
		}
		std::partial_sort(bottom_data_vector.begin(),
				bottom_data_vector.begin() + top_k_, bottom_data_vector.end(),
				std::greater<std::pair<Dtype, int> >());

		for (int k = 0; k < top_k_; ++k) {
			if (bottom_data_vector[k].second == static_cast<int>(bottom_label[i])) {
				++accuracy;
				break;
			}
		}
		prob_data += bottom[0]->offset(1);
	}
	top[0]->mutable_cpu_data()[0] = accuracy / num;
}
INSTANTIATE_CLASS(SpatialProbAggregationLayer);
REGISTER_LAYER_CLASS(SpatialProbAggregation);
} // namespace caffe

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using namespace cv;
namespace caffe {

template<typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top_k_ = this->layer_param_.accuracy_param().top_k();

	has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
	if (has_ignore_label_) {
		ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
	}

	denominator_ = this->layer_param_.accuracy_param().denominator();
	CHECK_GE(denominator_, 0)<< "Denominator must be positive; or 0, for the batch size.";

	rng_.reset(new Caffe::RNG(1013));
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())<< "top_k must be less than or equal to the number of classes.";
	label_axis_ =
	bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
	outer_num_ = bottom[0]->count(0, label_axis_);
	inner_num_ = bottom[0]->count(label_axis_ + 1);
	CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
	<< "Number of labels must match number of predictions; "
	<< "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
	<< "label count (number of labels) must be N*H*W, "
	<< "with integer values in {0, 1, ..., C-1}.";
	vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
	top[0]->Reshape(top_shape);
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int num = bottom[2]->num();
	const int img_ch = bottom[2]->channels();
	const int img_h = bottom[2]->height();
	const int img_w = bottom[2]->width();

	Dtype accuracy = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const int dim = bottom[0]->count() / outer_num_;
	const int num_labels = bottom[0]->shape(label_axis_);
	vector<Dtype> maxval(top_k_ + 1);
	vector<int> max_id(top_k_ + 1);
	int count = 0;

	vector<int> rand_nums;
	for (int i = 0; i < outer_num_; ++i) {
		caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
		int rand_num = ((*rng)() % 10000);
		rand_nums.push_back(rand_num);
		LOG(WARNING)<<"rand num "<<rand_num;
	}

	LOG(WARNING)<<"AccuracyLayer<Dtype>::Forward_cpu outer_num_ "
			<<outer_num_<<" inner_num_ "<<inner_num_;

	for (int i = 0; i < outer_num_; ++i) {
		vector<int> correct_index;
		for (int j = 0; j < inner_num_; ++j) {
			const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
			if(label_value < 10){
				correct_index.push_back(j);
			}

			if (has_ignore_label_ && label_value == ignore_label_) {
				continue;
			}
			CHECK_GE(label_value, 0);
			CHECK_LT(label_value, num_labels);
			// Top-k accuracy
			std::vector<std::pair<Dtype, int> > bottom_data_vector;
			for (int k = 0; k < num_labels; ++k) {
				bottom_data_vector.push_back(
						std::make_pair(bottom_data[i * dim + k * inner_num_ + j], k));
				DLOG(WARNING)<<"bottom_data "<<(i * dim + k * inner_num_ + j)
				<<" "<<bottom_data[i * dim + k * inner_num_ + j];
			}
			std::partial_sort(bottom_data_vector.begin(),
					bottom_data_vector.begin() + top_k_, bottom_data_vector.end(),
					std::greater<std::pair<Dtype, int> >());
			// check if true label is in top k predictions
			for (int k = 0; k < top_k_; k++) {
				if (bottom_data_vector[k].second == label_value) {
					++accuracy;
					if(label_value < 10){
//						correct_index.push_back(j);
					}
					break;
				}
			}
			++count;
		}

		Mat label_cv_img(img_w, img_h, CV_8UC3);
		for (int j = 0; j < correct_index.size(); ++j) {
			int h = correct_index[j] / img_w;
			int w = correct_index[j] % img_w;
			uchar* ptr = label_cv_img.ptr(h);
			ptr[3*w]=static_cast<unsigned char>(255);
			ptr[3*w+1]=static_cast<unsigned char>(255);
			ptr[3*w+2]=static_cast<unsigned char>(255);
		}
		char img_name[1024];
		int len = sprintf(img_name, "%d_label.bmp", rand_nums[i]);
		LOG(WARNING)<<"label image name: "<<img_name;
		imwrite(img_name, label_cv_img);
	}
	DLOG(WARNING)<<"AccuracyLayer Forward_cpu accuracy "<<accuracy
	<<" count "<<count<<" top_k_ "<<top_k_;

	// LOG(INFO) << "Accuracy: " << accuracy;
	const Dtype denominator = (denominator_ == 0) ? count : denominator_;
	top[0]->mutable_cpu_data()[0] = accuracy / denominator;
	// Accuracy layer should not be used as a loss function.

	// save input image into data
	const Dtype* img_data = bottom[2]->cpu_data();
	CHECK_EQ(img_ch, 1);

	int img_data_index = 0;
	for (int i = 0; i < num; ++i) {
		Mat cv_img(img_w, img_h, CV_8UC3);
		for (int h = 0; h < img_h; ++h) {
			uchar* ptr = cv_img.ptr(h);
			int cv_img_index = 0;
			for (int w = 0; w < img_w; ++w) {
				unsigned char pix_gray = static_cast<unsigned char>(128+img_data[img_data_index++]);
				for(int ch=0;ch<3;++ch){
					ptr[cv_img_index++] =pix_gray;
				}
			}
		}

		char img_name[1024];
		int len = sprintf(img_name, "%d_img.bmp", rand_nums[i]);
		LOG(WARNING)<<"saved image name: "<<img_name;
		imwrite(img_name, cv_img);
	}
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe

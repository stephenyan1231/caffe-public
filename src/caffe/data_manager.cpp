#include <opencv2/core/core.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_manager.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
BaseDataManager<Dtype>::BaseDataManager(const LayerParameter& data_layer_param,
		Net<Dtype> *net) :
		InternalThread(), layer_param_(data_layer_param), net_(net) {
}

template<typename Dtype>
BaseDataManager<Dtype>::~BaseDataManager() {
	JoinPrefetchThread();
}

// consider using forward_count_mutex_.lock/unlock when calling CreatePrefetchThread
template<typename Dtype>
void BaseDataManager<Dtype>::CreatePrefetchThread() {
	CreatePrefetchThread_();
	forward_count_ = 0;
	CHECK(StartInternalThread()) << "Thread execution failed";
}

template<typename Dtype>
void BaseDataManager<Dtype>::JoinPrefetchThread() {
	CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template<typename Dtype>
void BaseDataManager<Dtype>::SetBatchSize(int total_batch_size) {
	int replica_batch_size = divide_up(total_batch_size, Caffe::GetReplicasNum());
	for (int i = 0; i < Caffe::GetReplicasNum(); ++i) {
		int rest_size = total_batch_size - i * replica_batch_size;
		int this_replica_batch_size = std::min(replica_batch_size, rest_size);
		this->net_->SetBatchSize(i, this_replica_batch_size);
	}
}

INSTANTIATE_CLASS(BaseDataManager);

template<typename Dtype>
DataManager<Dtype>::DataManager(const LayerParameter& data_layer_param,
		Net<Dtype> *net) :
		BaseDataManager<Dtype>(data_layer_param, net), transform_param_(
				data_layer_param.transform_param()), data_transformer_(transform_param_,
				data_layer_param.phase()),  selective_list_fn_(data_layer_param.data_param().selective_list()){
	this->SetBatchSize(this->layer_param_.data_param().batch_size());
	// Hack
	if (data_layer_param.top_size() > 1) {
		this->output_labels_ = true;
	} else {
		this->output_labels_ = false;
	}

	if(selective_list_fn_ != std::string("")){
		LOG(INFO)<<"use selective list file "<<selective_list_fn_;
		std::ifstream selective_list_f(selective_list_fn_.c_str());
		CHECK(selective_list_f.is_open());
		std::string line;
		SelectiveItem item;
		while(std::getline(selective_list_f, line)){
		  std::stringstream ss(line);
		  ss>>item.img_name>>item.label;
		  selective_list_.push_back(item);
		}
		LOG(INFO)<<"selective list length "<<selective_list_.size();
		selective_list_cursor_ = 0;
		if(this->layer_param_.data_param().selective_list_shuffle()){
			shuffle(selective_list_.begin(), selective_list_.end());
		}
	}

	this->db_.reset(db::GetDB(data_layer_param.data_param().backend()));
	this->db_->Open(data_layer_param.data_param().source(), db::READ);
	LOG(INFO)<<"new database cursor";
	this->cursor_.reset(this->db_->NewCursor());
	LOG(INFO)<<"new database transaction";
	this->transaction_.reset(this->db_->NewTransaction(true));
	LOG(INFO)<<"[[[[[database key "<<this->cursor_->key()<<" "
			<<this->cursor_->key().length();

	// Check if we should randomly skip a few data points
	if (data_layer_param.data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
				% data_layer_param.data_param().rand_skip();
		LOG(INFO)<< "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			this->cursor_->Next();
		}
	}

	// Read a data point, and use it to initialize the top blob.
	Datum datum;
	datum.ParseFromString(this->cursor_->value());

	if (DecodeDatum(&datum)) {
		LOG(INFO)<< "Decoding Datum";
	}
	this->datum_channels_ = datum.channels();
	this->datum_height_ = datum.height();
	this->datum_width_ = datum.width();

	// image
	int crop_size = data_layer_param.transform_param().crop_size();
	if (crop_size > 0) {
		this->prefetch_data_.Reshape(data_layer_param.data_param().batch_size(),
				datum.channels(), crop_size, crop_size);
		this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
	} else {
		this->prefetch_data_.Reshape(data_layer_param.data_param().batch_size(),
				datum.channels(), datum.height(), datum.width());
		this->transformed_data_.Reshape(1, datum.channels(), datum.height(),
				datum.width());
	}
	LOG(INFO)<< "DataManager: prefetch data size: " << this->prefetch_data_.num()
	<< "," << this->prefetch_data_.channels() << ","
	<< this->prefetch_data_.height() << "," << this->prefetch_data_.width();
	// label
	if (this->output_labels_) {
		this->prefetch_label_.Reshape(data_layer_param.data_param().batch_size(), 1,
				1, 1);
	}
	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
	this->prefetch_data_.mutable_cpu_data();
	if (this->output_labels_) {
		this->prefetch_label_.mutable_cpu_data();
	}
}

template<typename Dtype>
DataManager<Dtype>::~DataManager() {

}

template<typename Dtype>
void DataManager<Dtype>::InternalThreadEntry() {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(this->prefetch_data_.count());
	CHECK(this->transformed_data_.count());
	if (this->output_labels_) {
		CHECK(this->prefetch_label_.count());
	}
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

	if (this->output_labels_) {
		top_label = this->prefetch_label_.mutable_cpu_data();
	}
	const int batch_size = this->layer_param_.data_param().batch_size();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		timer.Start();
		// get a blob
		Datum datum;
		int item_label = -1;
		if(selective_list_.size() > 0){
			std::string item_img_name = selective_list_[selective_list_cursor_].img_name;
			item_label = selective_list_[selective_list_cursor_].label;
			datum.ParseFromString(this->transaction_->GetValue(item_img_name));
			datum.set_label(item_label);
			selective_list_cursor_++;
			if(selective_list_cursor_ == selective_list_.size()){
				selective_list_cursor_ = 0;
				if(this->layer_param_.data_param().selective_list_shuffle()){
					shuffle(selective_list_.begin(), selective_list_.end());
				}
			}
		}
		else{
			datum.ParseFromString(this->cursor_->value());
		}

		cv::Mat cv_img;
		if (datum.encoded()) {
			cv_img = DecodeDatumToCVMat(datum);
		}
		read_time += timer.MicroSeconds();
		timer.Start();

		// Apply data transformations (mirror, scale, crop...)
		int offset = this->prefetch_data_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset);
		if (datum.encoded()) {
			this->data_transformer_.Transform(cv_img, &(this->transformed_data_));
		} else {
			this->data_transformer_.Transform(datum, &(this->transformed_data_));
		}
		if (this->output_labels_) {
			top_label[item_id] = datum.label();
		}
		trans_time += timer.MicroSeconds();

		if(selective_list_.size() == 0){
			// go to the next iter
			this->cursor_->Next();
			if (!this->cursor_->valid()) {
				DLOG(INFO)<< "Restarting data prefetching from start.";
				this->cursor_->SeekToFirst();
			}
		}
	}
	batch_timer.Stop();

	DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void DataManager<Dtype>::CopyFetchDataToConvThread(int replica_id,
		const vector<Blob<Dtype>*>& top) {

	this->forward_count_mutex_.lock();
	this->forward_count_++;
	if (this->forward_count_ == 1) {
		// First, join the thread
		this->JoinPrefetchThread();
//		JoinPrefetchThread();
	}
	this->forward_count_mutex_.unlock();
	int num_replicas = Caffe::GetReplicasNum();
	int batch_size = prefetch_data_.num();
	int replica_batch_size = divide_up(batch_size, num_replicas);
	int start = replica_batch_size * replica_id;
	int end = start + this->net_->GetBatchSize(replica_id);

//	CHECK_EQ(top[0]->num(), end - start);
//	CHECK_EQ(top[0]->channels(), prefetch_data_.channels());
//	CHECK_EQ(top[0]->height(), prefetch_data_.height());
//	CHECK_EQ(top[0]->width(), prefetch_data_.width());
	top[0]->Reshape(end-start,prefetch_data_.channels(),prefetch_data_.height(),prefetch_data_.width());

	int unit_size = prefetch_data_.count() / prefetch_data_.num();

	if (Caffe::mode() == Caffe::CPU) {
		caffe_copy(unit_size * (end - start),
				prefetch_data_.cpu_data() + prefetch_data_.offset(start),
				top[0]->mutable_cpu_data());
	} else {
		caffe_copy(unit_size * (end - start),
				prefetch_data_.cpu_data() + prefetch_data_.offset(start),
				top[0]->mutable_gpu_data());
	}

	if (output_labels_) {
		CHECK_EQ(top[1]->num(), end - start);
		CHECK_EQ(top[1]->channels(), prefetch_label_.channels());
		CHECK_EQ(top[1]->height(), prefetch_label_.height());
		CHECK_EQ(top[1]->width(), prefetch_label_.width());
		if (Caffe::mode() == Caffe::CPU) {
			caffe_copy(end - start,
					prefetch_label_.cpu_data() + prefetch_label_.offset(start),
					top[1]->mutable_cpu_data());
		} else {
			caffe_copy(end - start,
					prefetch_label_.cpu_data() + prefetch_label_.offset(start),
					top[1]->mutable_gpu_data());
		}
	}

	this->forward_count_mutex_.lock();
	if (this->forward_count_ == num_replicas) {
		// create thread to fetch next batch data
		this->CreatePrefetchThread();
	}
	this->forward_count_mutex_.unlock();
}

// consider using forward_count_mutex_.lock/unlock when calling CreatePrefetchThread
template<typename Dtype>
void DataManager<Dtype>::CreatePrefetchThread_() {
	this->data_transformer_.InitRand();
}

INSTANTIATE_CLASS(DataManager);

template<typename Dtype>
DataVariableSizeManager<Dtype>::DataVariableSizeManager(
		const LayerParameter& data_layer_param, Net<Dtype> *net) :
		BaseDataManager<Dtype>(data_layer_param, net), transform_param_(
				data_layer_param.transform_param()), data_transformer_(transform_param_,
				data_layer_param.phase()),
				selective_list_fn_(data_layer_param.data_variable_size_param().selective_list()) {
	DataVariableSizeParameter data_variable_size_param =
			data_layer_param.data_variable_size_param();
	this->SetBatchSize(data_variable_size_param.batch_size());
	// Hack
	if (data_layer_param.top_size() > 1) {
		this->output_labels_ = true;
	} else {
		this->output_labels_ = false;
	}

	if(selective_list_fn_ != std::string("")){
		LOG(INFO)<<"use selective list file "<<selective_list_fn_;
		std::ifstream selective_list_f(selective_list_fn_.c_str());
		CHECK(selective_list_f.is_open());
		std::string line;
		SelectiveItem item;
		while(std::getline(selective_list_f, line)){
		  std::stringstream ss(line);
		  ss>>item.img_name>>item.label;
		  selective_list_.push_back(item);
		}
		LOG(INFO)<<"selective list length "<<selective_list_.size();
		selective_list_cursor_ = 0;
		if(data_variable_size_param.selective_list_shuffle()){
			shuffle(selective_list_.begin(), selective_list_.end());
		}
	}

	datum_max_pixel_num_ = data_variable_size_param.max_pixel_num();

	this->db_.reset(
			db::GetDB(data_layer_param.data_variable_size_param().backend()));
	this->db_->Open(data_layer_param.data_variable_size_param().source(),
			db::READ);
	this->cursor_.reset(this->db_->NewCursor());
	this->transaction_.reset(this->db_->NewTransaction(true));

	// Check if we should randomly skip a few data points
	if (data_layer_param.data_variable_size_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
				% data_layer_param.data_variable_size_param().rand_skip();
		LOG(INFO)<< "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			this->cursor_->Next();
		}
	}

	// Read a data point, and use it to initialize the top blob.
	Datum datum;
	datum.ParseFromString(this->cursor_->value());
	if (DecodeDatum(&datum)) {
		LOG(INFO)<< "Decoding Datum";
	}

	this->datum_channels_ = datum.channels();

	// image
	this->prefetch_data_.Reshape(data_variable_size_param.batch_size(),
			datum.channels(), 1, datum_max_pixel_num_);
	prefetch_data_size_.Reshape(data_variable_size_param.batch_size(), 1, 1, 2);
	this->transformed_data_.Reshape(1, datum.channels(), 1, datum_max_pixel_num_);

	LOG(INFO)<< "DataVariableSizeManager: prefetch data size: " << this->prefetch_data_.num()
	<< "," << this->prefetch_data_.channels() << ","
	<< this->prefetch_data_.height() << "," << this->prefetch_data_.width();
	// label
	if (this->output_labels_) {
		this->prefetch_label_.Reshape(data_variable_size_param.batch_size(), 1, 1,
				1);
	}
	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
	this->prefetch_data_.mutable_cpu_data();
	if (this->output_labels_) {
		this->prefetch_label_.mutable_cpu_data();
	}

	replicas_batch_data_max_size_.Reshape(Caffe::GetReplicasNum(), 1, 1, 2);
	prefetch_data_reorganized_.resize(Caffe::GetReplicasNum());
	for (int i = 0; i < Caffe::GetReplicasNum(); ++i) {
		prefetch_data_reorganized_[i] = new Blob<Dtype>();
	}
}

template<typename Dtype>
DataVariableSizeManager<Dtype>::~DataVariableSizeManager() {
	for (int i = 0; i < Caffe::GetReplicasNum(); ++i) {
		delete prefetch_data_reorganized_[i];
	}
}

template<typename Dtype>
void DataVariableSizeManager<Dtype>::InternalThreadEntry() {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(this->prefetch_data_.count());
	CHECK(this->transformed_data_.count());
	if (this->output_labels_) {
		CHECK(this->prefetch_label_.count());
	}
	Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* prefetch_label = NULL; // suppress warnings about uninitialized variables

	if (this->output_labels_) {
		prefetch_label = this->prefetch_label_.mutable_cpu_data();
	}
	const int batch_size =
			this->layer_param_.data_variable_size_param().batch_size();
	Dtype *prefetch_data_size = prefetch_data_size_.mutable_cpu_data();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		timer.Start();
		// get a blob
		Datum datum;
		if(selective_list_.size() > 0){
			std::string item_img_name = selective_list_[selective_list_cursor_].img_name;
			int item_label = selective_list_[selective_list_cursor_].label;
			datum.ParseFromString(this->transaction_->GetValue(item_img_name));
			datum.set_label(item_label);
			selective_list_cursor_++;
			if(selective_list_cursor_ == selective_list_.size()){
				selective_list_cursor_ = 0;
				if(this->layer_param_.data_variable_size_param().selective_list_shuffle()){
					shuffle(selective_list_.begin(), selective_list_.end());
				}
			}
		}else{
			LOG(INFO)<<"get item from database with key "<<this->cursor_->key();
			datum.ParseFromString(this->cursor_->value());
		}

		cv::Mat cv_img;
		if (datum.encoded()) {
			cv_img = DecodeDatumToCVMat(datum);
		}
		read_time += timer.MicroSeconds();
		timer.Start();

		// Apply data transformations (mirror, scale, crop...)
		int offset = prefetch_data_.offset(item_id);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);
		int datum_height = 0, datum_width = 0;
		if (datum.encoded()) {
			this->data_transformer_.Transform(cv_img, &(this->transformed_data_), datum_height,
					datum_width);
		} else {
			this->data_transformer_.Transform(datum, &(this->transformed_data_),
					datum_height, datum_width);
		}
		(prefetch_data_size + prefetch_data_size_.offset(item_id))[0] = datum_height;
		(prefetch_data_size + prefetch_data_size_.offset(item_id))[1] = datum_width;

		if (this->output_labels_) {
			prefetch_label[item_id] = datum.label();
		}
		trans_time += timer.MicroSeconds();

		// go to the next iter
		this->cursor_->Next();
		if (!this->cursor_->valid()) {
			DLOG(INFO)<< "Restarting data prefetching from start.";
			this->cursor_->SeekToFirst();
		}
	}
	batch_timer.Stop();

	int num_replicas = Caffe::GetReplicasNum();
	int replica_batch_size = divide_up(batch_size, num_replicas);
	Dtype* replicas_batch_data_max_size =
			replicas_batch_data_max_size_.mutable_cpu_data();
	for (int i = 0; i < num_replicas; ++i) {
		int start = replica_batch_size * i;
		int end = start + this->net_->GetBatchSize(i);
		int max_height = 0, max_width = 0;
		for (int j = start; j < end; ++j) {
			int height = (prefetch_data_size + prefetch_data_size_.offset(j))[0];
			int width = (prefetch_data_size + prefetch_data_size_.offset(j))[1];
			max_height = max_height > height ? max_height : height;
			max_width = max_width > width ? max_width : width;
		}
		(replicas_batch_data_max_size + replicas_batch_data_max_size_.offset(i))[0] =
				max_height;
		(replicas_batch_data_max_size + replicas_batch_data_max_size_.offset(i))[1] =
				max_width;
		/* re-organize memory layout in prefetch_data so that the first (channels*max_height*max_width) data are
		 * initialized properly
		 * */
//		LOG(INFO)<<"DataVariableSizeManager<Dtype>::InternalThreadEntry max_height "<<max_height<<
//		" max_width "<<max_width<<" count "<<
//		(end-start)*this->datum_channels_*max_height*max_width;
		prefetch_data_reorganized_[i]->Reshape(end - start, this->datum_channels_,
				max_height, max_width);
		Dtype* prefetch_data_reorgnized =
				prefetch_data_reorganized_[i]->mutable_cpu_data();
		caffe_memset(
				sizeof(Dtype) * prefetch_data_reorganized_[i]->num()
						* prefetch_data_reorganized_[i]->channels()
						* prefetch_data_reorganized_[i]->height()
						* prefetch_data_reorganized_[i]->width(), 0,
				prefetch_data_reorgnized);
		for (int j = start; j < end; ++j) {
			int height = (prefetch_data_size + prefetch_data_size_.offset(j))[0];
			int width = (prefetch_data_size + prefetch_data_size_.offset(j))[1];
			Dtype * data_ptr = prefetch_data + prefetch_data_.offset(j);
			for (int index = 0, c = 0; c < this->datum_channels_; ++c) {
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w, ++index) {
						int index2 = (c * max_height + h) * max_width + w;
						(prefetch_data_reorgnized
								+ prefetch_data_reorganized_[i]->offset(j - start))[index2] =
								data_ptr[index];
					}
				}
			}
		}
	}

	DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void DataVariableSizeManager<Dtype>::CopyFetchDataToConvThread(int replica_id,
		const vector<Blob<Dtype>*>& top) {

	this->forward_count_mutex_.lock();
	this->forward_count_++;
	if (this->forward_count_ == 1) {
		// First, join the thread
		this->JoinPrefetchThread();
//		JoinPrefetchThread();
	}
	this->forward_count_mutex_.unlock();
	int num_replicas = Caffe::GetReplicasNum();
	int batch_size = prefetch_data_.num();
	int replica_batch_size = divide_up(batch_size, num_replicas);
	int start = replica_batch_size * replica_id;
	int end = start + this->net_->GetBatchSize(replica_id);

	top[0]->ReshapeLike(*prefetch_data_reorganized_[replica_id]);

	if (Caffe::mode() == Caffe::CPU) {
		caffe_copy(prefetch_data_reorganized_[replica_id]->count(),
				prefetch_data_reorganized_[replica_id]->cpu_data(),
				top[0]->mutable_cpu_data());

		caffe_copy(2 * (end - start),
				prefetch_data_size_.cpu_data() + prefetch_data_size_.offset(start),
				top[1]->mutable_cpu_data());

	} else {

		caffe_copy(prefetch_data_reorganized_[replica_id]->count(),
				prefetch_data_reorganized_[replica_id]->cpu_data(),
				top[0]->mutable_gpu_data());

		caffe_copy(2 * (end - start),
				prefetch_data_size_.cpu_data() + prefetch_data_size_.offset(start),
				top[1]->mutable_gpu_data());
	}

	if (output_labels_) {
		CHECK_EQ(top[2]->num(), end - start);
		CHECK_EQ(top[2]->channels(), prefetch_label_.channels());
		CHECK_EQ(top[2]->height(), prefetch_label_.height());
		CHECK_EQ(top[2]->width(), prefetch_label_.width());
		if (Caffe::mode() == Caffe::CPU) {
			caffe_copy(end - start,
					prefetch_label_.cpu_data() + prefetch_label_.offset(start),
					top[2]->mutable_cpu_data());
		} else {
			caffe_copy(end - start,
					prefetch_label_.cpu_data() + prefetch_label_.offset(start),
					top[2]->mutable_gpu_data());
		}
	}

	this->forward_count_mutex_.lock();
	if (this->forward_count_ == num_replicas) {
		// create thread to fetch next batch data
		this->CreatePrefetchThread();
	}
	this->forward_count_mutex_.unlock();
}

// consider using forward_count_mutex_.lock/unlock when calling CreatePrefetchThread
template<typename Dtype>
void DataVariableSizeManager<Dtype>::CreatePrefetchThread_() {
	this->data_transformer_.InitRand();
}

INSTANTIATE_CLASS(DataVariableSizeManager);
}  // namespace caffe

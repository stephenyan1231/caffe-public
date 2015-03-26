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

INSTANTIATE_CLASS(BaseDataManager);

template<typename Dtype>
DataManager<Dtype>::DataManager(const LayerParameter& data_layer_param,
		Net<Dtype> *net) :
		BaseDataManager<Dtype>(data_layer_param, net), transform_param_(
				data_layer_param.transform_param()),
				data_transformer_(transform_param_, data_layer_param.phase()) {
	// Hack
	if (data_layer_param.top_size() > 1) {
		this->output_labels_ = true;
	} else {
		this->output_labels_ = false;
	}

	this->db_.reset(db::GetDB(data_layer_param.data_param().backend()));
	this->db_->Open(data_layer_param.data_param().source(), db::READ);
	this->cursor_.reset(this->db_->NewCursor());

	// Check if we should randomly skip a few data points
	if (data_layer_param.data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
				% data_layer_param.data_param().rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			this->cursor_->Next();
		}
	}

	// Read a data point, and use it to initialize the top blob.
	Datum datum;
	datum.ParseFromString(this->cursor_->value());

	if (DecodeDatum(&datum)) {
		LOG(INFO) << "Decoding Datum";
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
	LOG(INFO) << "DataManager: prefetch data size: " << this->prefetch_data_.num()
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
		datum.ParseFromString(this->cursor_->value());

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

		// go to the next iter
		this->cursor_->Next();
		if (!this->cursor_->valid()) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			this->cursor_->SeekToFirst();
		}
	}
	batch_timer.Stop();

	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
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
//	prefetch_data_mutex_.lock_shared();
	int batch_size = prefetch_data_.num();
	int replica_batch_size = divide_up(batch_size, num_replicas);
	int start = replica_batch_size * replica_id;
	int end = start + this->net_->GetBatchSize(replica_id);

	CHECK_EQ(top[0]->num(), end - start);
	CHECK_EQ(top[0]->channels(), prefetch_data_.channels());
	CHECK_EQ(top[0]->height(), prefetch_data_.height());
	CHECK_EQ(top[0]->width(), prefetch_data_.width());

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
}  // namespace caffe

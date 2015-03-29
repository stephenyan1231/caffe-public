#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/data_manager.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
}

template<typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
//	int num_replicas = Caffe::GetReplicasNum();
//	int replica_batch_size = divide_up(
//			this->layer_param_.data_param().batch_size(), num_replicas);
//	int rest_size = this->layer_param_.data_param().batch_size() - this->replica_id_ * replica_batch_size;
//	int this_replica_batch_size = std::min(replica_batch_size, rest_size);
//	this->net_->SetBatchSize(this->replica_id_, this_replica_batch_size);

	int this_replica_batch_size = this->net_->GetBatchSize(this->replica_id_);

	int datum_channels = this->net_->GetDataManager()->GetDatumChannels();
	int datum_height = this->net_->GetDataManager()->GetDatumHeight();
	int datum_width = this->net_->GetDataManager()->GetDatumWidth();
	// image
	int crop_size = this->layer_param_.transform_param().crop_size();
	if (crop_size > 0) {
		top[0]->Reshape(this_replica_batch_size,
				datum_channels, crop_size, crop_size);
	} else {
		top[0]->Reshape(this_replica_batch_size,
				datum_channels, datum_height, datum_width);
	}
	LOG(INFO)<< "output data size: " << top[0]->num() << ","
	<< top[0]->channels() << "," << top[0]->height() << ","
	<< top[0]->width();
	// label
	if (this->output_labels_) {
		top[1]->Reshape(this_replica_batch_size, 1, 1, 1);
	}
}

template<typename Dtype>
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	BaseDataManager<Dtype> *dm = this->net_->GetDataManager();
	dm->CopyFetchDataToConvThread(this->replica_id_, top);
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe

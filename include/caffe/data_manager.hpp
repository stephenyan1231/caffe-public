#ifndef CAFFE_DATA_MANAGER_HPP_
#define CAFFE_DATA_MANAGER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template<typename Dtype>
class Net;

template<typename Dtype>
class BaseDataManager: public InternalThread {
public:
	explicit BaseDataManager(const LayerParameter& data_layer_param,
			Net<Dtype> *net);
	~BaseDataManager();

	inline int GetDatumChannels() {
		return datum_channels_;
	}
	inline int GetDatumHeight() {
		return datum_height_;
	}
	inline int GetDatumWidth() {
		return datum_width_;
	}

	virtual void CreatePrefetchThread();
	virtual void JoinPrefetchThread();
	virtual void InternalThreadEntry() = 0;
	virtual void CopyFetchDataToConvThread(int replica_id,
			const vector<Blob<Dtype>*>& top) = 0;

protected:
	virtual void CreatePrefetchThread_() = 0;

	LayerParameter layer_param_;
	Net<Dtype> *net_;

	shared_ptr<db::DB> db_;
	shared_ptr<db::Cursor> cursor_;

	int datum_channels_, datum_height_, datum_width_;

	int forward_count_;
	boost::mutex forward_count_mutex_;
};

template<typename Dtype>
class DataManager: public BaseDataManager<Dtype> {
public:
	explicit DataManager(const LayerParameter& data_layer_param, Net<Dtype> *net);
	~DataManager();

//	virtual void JoinPrefetchThread();
	virtual void InternalThreadEntry();
	virtual void CopyFetchDataToConvThread(int replica_id,
			const vector<Blob<Dtype>*>& top);

protected:
	virtual void CreatePrefetchThread_();

	Blob<Dtype> prefetch_data_;
	Blob<Dtype> prefetch_label_;
	Blob<Dtype> transformed_data_;

//	int datum_channels_, datum_height_, datum_width_;

	bool output_labels_;
	TransformationParameter transform_param_;
	DataTransformer<Dtype> data_transformer_;

	boost::shared_mutex prefetch_data_mutex_;

};

}  // namespace caffe

#endif  // CAFFE_DATA_MANAGER_HPP_

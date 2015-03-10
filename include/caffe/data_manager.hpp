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

template <typename Dtype>
class Net;

template <typename Dtype>
class DataManager :public InternalThread{
public:
	explicit DataManager(const LayerParameter& data_layer_param, Net<Dtype> *net);
	~DataManager();

	virtual void CreatePrefetchThread();
	virtual void JoinPrefetchThread();
	virtual void InternalThreadEntry();
	void CopyFetchDataToConvThread(int replica_id, const vector<Blob<Dtype>*>& top);

	inline int GetDatumChannels(){return datum_channels_;}
	inline int GetDatumHeight(){return datum_height_;}
	inline int GetDatumWidth(){return datum_width_;}

protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
  Blob<Dtype> transformed_data_;

//  Blob<Dtype> fetch_data_;
//  Blob<Dtype> fetch_label_;


  LayerParameter layer_param_;
  bool output_labels_;
	TransformationParameter transform_param_;
	DataTransformer<Dtype> data_transformer_;


  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;

  Net<Dtype> *net_;

  int datum_channels_, datum_height_,datum_width_;

  int forward_count_;

  boost::shared_mutex prefetch_data_mutex_;
  boost::mutex forward_count_mutex_;

};


}  // namespace caffe

#endif  // CAFFE_DATA_MANAGER_HPP_

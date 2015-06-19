#ifndef CAFFE_DATA_LAYERS_MORE_HPP_
#define CAFFE_DATA_LAYERS_MORE_HPP_

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class SemanticLabelingDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SemanticLabelingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SemanticLabelingDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SemanticLabelingData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
};

} // namespace caffe

#endif // CAFFE_DATA_LAYERS_MORE_HPP_

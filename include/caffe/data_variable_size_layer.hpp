#ifndef CAFFE_DATA_VARIABLE_SIZE_LAYER_HPP_
#define CAFFE_DATA_VARIABLE_SIZE_LAYER_HPP_

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class DataVariableSizeLayer : public BaseDataLayer<Dtype> {
 public:
  explicit DataVariableSizeLayer(const LayerParameter& param, int replica_id, Net<Dtype> *net)
      : BaseDataLayer<Dtype>(param,replica_id,net) {}
  virtual ~DataVariableSizeLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  			const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DataVariableSize"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  /* 1st output: image pixel
   * 2nd output: image size (height, width)
   * 3rd output: image label
  */
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
};

} // namespace caffe


#endif // CAFFE_DATA_VARIABLE_SIZE_LAYER_HPP_

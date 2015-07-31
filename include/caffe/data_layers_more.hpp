#ifndef CAFFE_DATA_LAYERS_MORE_HPP_
#define CAFFE_DATA_LAYERS_MORE_HPP_

#include "caffe/data_layers.hpp"
#include "caffe/semantic_labeling_data_transformer.hpp"

namespace caffe {

template <typename Dtype>
class SemanticLabelingDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SemanticLabelingDataLayer(const LayerParameter& param);
  virtual ~SemanticLabelingDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SemanticLabelingData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

//  void Fill_label_weight_map(const Blob<Dtype> &label_map, Blob<Dtype> &label_weight_map);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 protected:
  virtual void InternalThreadEntry();

  SemanticLabelingTransformationParameter semantic_labeling_transform_param_;
  shared_ptr<SemanticLabelingDataTransformer<Dtype> > semantic_labeling_data_transformer_;
  Blob<Dtype> transformed_label_;

//  bool output_label_weight_map_;
//  Blob<Dtype> prefetch_label_weight_map_;

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
//  std::vector<Dtype> label_weights_;
};

/**
 * @brief Image and segmentation pair data provider.
 * Image sizes are uniform within the mini-batch
 * OUTPUT:
 * 0: (num, channels, height, width): image values
 * 1: (num, 1, height, width): labels
 */
template <typename Dtype>
class ImageSegUniformSizeDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegUniformSizeDataLayer(const LayerParameter& param);
  virtual ~ImageSegUniformSizeDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegUniformSizeData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 protected:
  virtual void InternalThreadEntry();

  Blob<Dtype> transformed_label_;
  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
};

} // namespace caffe

#endif // CAFFE_DATA_LAYERS_MORE_HPP_

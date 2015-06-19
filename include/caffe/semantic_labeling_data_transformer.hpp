#ifndef CAFFE_SEMANTIC_LABELING_DATA_TRANSFORMER_HPP
#define CAFFE_SEMANTIC_LABELING_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class SemanticLabelingDataTransformer {
 public:
  explicit SemanticLabelingDataTransformer(
  		const SemanticLabelingTransformationParameter& param, Phase phase);
  virtual ~SemanticLabelingDataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const SemanticLabelingDatum& datum, Blob<Dtype>* transformed_blob,
  		Blob<Dtype>* transformed_label);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const SemanticLabelingDatum& datum, const cv::Mat& cv_img,
  		Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label);

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  void Transform(const SemanticLabelingDatum& datum, Dtype* transformed_data,
  		Dtype* transformed_label);
  // Tranformation parameters
  SemanticLabelingTransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_SEMANTIC_LABELING_DATA_TRANSFORMER_HPP


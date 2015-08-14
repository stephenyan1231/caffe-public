#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/dimension_transpose_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template<typename TypeParam>
class DimensionTransposeLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DimensionTransposeLayerTest() :
      blob_bottom_(new Blob<Dtype>(2, 2, 4, 4)), blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DimensionTransposeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void InitParam(DimensionTransposeParameter *dimension_transpose_param,
      DimensionTransposeParameter::Direction dir) {
    dimension_transpose_param->set_direction(dir);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DimensionTransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(DimensionTransposeLayerTest, TestGradient_Num_First_To_Height_First) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DimensionTransposeParameter *dimension_transpose_param =
      layer_param.mutable_dimension_transpose_param();
  this->InitParam(dimension_transpose_param,
      DimensionTransposeParameter_Direction_NUM_FIRST_TO_HEIGHT_FIRST);

  DimensionTransposeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(DimensionTransposeLayerTest, TestGradient_Height_First_To_Num_First) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DimensionTransposeParameter *dimension_transpose_param =
      layer_param.mutable_dimension_transpose_param();
  this->InitParam(dimension_transpose_param,
      DimensionTransposeParameter_Direction_HEIGHT_FIRST_TO_NUM_FIRST);

  DimensionTransposeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  //  namespace caffe

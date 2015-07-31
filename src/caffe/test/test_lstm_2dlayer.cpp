#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/lstm_2dlayer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
class LSTM_2DLayerTest : public ::testing::Test {
 protected:
	LSTM_2DLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 8, 8)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LSTM_2DLayerTest() { delete blob_bottom_; delete blob_top_; }

  virtual void InitParam(LSTM2DParameter *lstm_2d_param, int num_output,
        int patch_width, int patch_height, Dtype forget_gate_scaling_factor = 0.5) {
    lstm_2d_param->set_num_output(num_output);
    lstm_2d_param->set_patch_width(patch_width);
    lstm_2d_param->set_patch_height(num_output);
    lstm_2d_param->set_forget_gate_scaling_factor(forget_gate_scaling_factor);

    lstm_2d_param->mutable_general_weight_filler()->set_type("uniform");
    lstm_2d_param->mutable_general_weight_filler()->set_min(-1);
    lstm_2d_param->mutable_general_weight_filler()->set_max(1);

    lstm_2d_param->mutable_general_bias_filler()->set_type("constant");
    lstm_2d_param->mutable_general_bias_filler()->set_value(0);

    lstm_2d_param->mutable_forget_gate_bias_filler()->set_type("constant");
    lstm_2d_param->mutable_forget_gate_bias_filler()->set_value(1.0);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LSTM_2DLayerTest, TestDtypes);

TYPED_TEST(LSTM_2DLayerTest, TestGradientPatch2x2){
  LayerParameter layer_param;
  LSTM2DParameter *lstm_2d_param = layer_param.mutable_lstm_2d_param();
  this->InitParam(lstm_2d_param, 2, 2, 2);
  LSTM_2DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTM_2DLayerTest, TestGradientPatch1x1){
  LayerParameter layer_param;
  LSTM2DParameter *lstm_2d_param = layer_param.mutable_lstm_2d_param();
  this->InitParam(lstm_2d_param, 2, 1, 1);
  LSTM_2DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}

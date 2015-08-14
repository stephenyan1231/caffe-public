#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/grid_lstm_3D_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename Dtype>
class GridLSTM3DLayerTest: public ::testing::Test {
 protected:
  GridLSTM3DLayerTest() :
      blob_bottom_cstate_(new Blob<Dtype>(4, 6, 2, 2)), blob_bottom_hidden_(
          new Blob<Dtype>(4, 6, 2, 2)), blob_top_cstate_(new Blob<Dtype>()),
          blob_top_hidden_(
          new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_cstate_);
    filler.Fill(blob_bottom_hidden_);
    blob_bottom_vec_.push_back(blob_bottom_cstate_);
    blob_bottom_vec_.push_back(blob_bottom_hidden_);
    blob_top_vec_.push_back(blob_top_cstate_);
    blob_top_vec_.push_back(blob_top_hidden_);
  }
  virtual ~GridLSTM3DLayerTest() {
    delete blob_bottom_cstate_;
    delete blob_bottom_hidden_;
    delete blob_top_cstate_;
    delete blob_top_hidden_;
  }

  virtual void InitParam(GridLSTM3DParameter *grid_lstm_3d_param,
      GridLSTM3DParameter::Direction x_dir,
      GridLSTM3DParameter::Direction y_dir, int num_output,
      bool peephole = true) {
    grid_lstm_3d_param->set_x_direction(x_dir);
    grid_lstm_3d_param->set_y_direction(y_dir);
    grid_lstm_3d_param->set_num_output(num_output);
    grid_lstm_3d_param->set_peephole(peephole);

    grid_lstm_3d_param->mutable_general_weight_filler()->set_type("uniform");
    grid_lstm_3d_param->mutable_general_weight_filler()->set_min(-0.2);
    grid_lstm_3d_param->mutable_general_weight_filler()->set_max(0.2);

    grid_lstm_3d_param->mutable_cell_input_bias_filler()->set_type("constant");
    grid_lstm_3d_param->mutable_cell_input_bias_filler()->set_value(0);

    grid_lstm_3d_param->mutable_forget_gate_bias_filler()->set_type(
        "constant");
    grid_lstm_3d_param->mutable_forget_gate_bias_filler()->set_value(1.0);

    grid_lstm_3d_param->mutable_input_gate_bias_filler()->set_type("constant");
    grid_lstm_3d_param->mutable_input_gate_bias_filler()->set_value(0.0);

    grid_lstm_3d_param->mutable_output_gate_bias_filler()->set_type(
        "constant");
    grid_lstm_3d_param->mutable_output_gate_bias_filler()->set_value(0.0);
  }

  Blob<Dtype>* const blob_bottom_cstate_;
  Blob<Dtype>* const blob_bottom_hidden_;
  Blob<Dtype>* const blob_top_cstate_;
  Blob<Dtype>* const blob_top_hidden_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GridLSTM3DLayerTest, TestDtypes);

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_PosX_PosY) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_POSITIVE,
      GridLSTM3DParameter_Direction_POSITIVE, 2, true);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_NegX_PosY) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_NEGATIVE,
      GridLSTM3DParameter_Direction_POSITIVE, 2, true);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_PosX_NegY) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_POSITIVE,
      GridLSTM3DParameter_Direction_NEGATIVE, 2, true);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_NegX_NegY) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_NEGATIVE,
      GridLSTM3DParameter_Direction_NEGATIVE, 2, true);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_PosX_PosY_NoPeephole) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_POSITIVE,
      GridLSTM3DParameter_Direction_POSITIVE, 2, false);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_NegX_PosY_NoPeephole) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_NEGATIVE,
      GridLSTM3DParameter_Direction_POSITIVE, 2, false);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_PosX_NegY_NoPeephole) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_POSITIVE,
      GridLSTM3DParameter_Direction_NEGATIVE, 2, false);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GridLSTM3DLayerTest, TestGradient_NegX_NegY_NoPeephole) {
  LayerParameter layer_param;
  GridLSTM3DParameter *grid_lstm_3d_param =
      layer_param.mutable_grid_lstm_3d_param();
  this->InitParam(grid_lstm_3d_param, GridLSTM3DParameter_Direction_NEGATIVE,
      GridLSTM3DParameter_Direction_NEGATIVE, 2, false);
  GridLSTM3DLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe

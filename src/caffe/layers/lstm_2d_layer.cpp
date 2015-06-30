#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_2d_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void LSTM2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	input_activation_func_ =
			this->layer_param_.lstm_2d_unit_param().input_activation();
	output_activation_func_ =
			this->layer_param_.lstm_2d_unit_param().output_activation();
	Recurrent2DLayer<Dtype>::LayerSetUp(bottom, top);
}

template<typename Dtype>
void LSTM2DLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
	const int num_output = this->layer_param_.recurrent_2d_param().num_output();
	CHECK_GT(num_output, 0);
	const FillerParameter& weight_filler =
			this->layer_param_.recurrent_2d_param().weight_filler();
	const FillerParameter& bias_filler =
			this->layer_param_.recurrent_2d_param().bias_filler();

	// Add generic LayerParameter's (without bottoms/tops) of layer types we'll
	// use to save redundant code.
	LayerParameter hidden_param;
	hidden_param.set_type("InnerProduct");
	hidden_param.mutable_inner_product_param()->set_num_output(num_output * 5);
	hidden_param.mutable_inner_product_param()->set_bias_term(false);
	hidden_param.mutable_inner_product_param()->set_axis(2);
	hidden_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(
			weight_filler);

	LayerParameter biased_hidden_param(hidden_param);
	biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
	biased_hidden_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(
			bias_filler);

	LayerParameter slice_param;
	slice_param.set_type("Slice");
	slice_param.mutable_slice_param()->set_axis(0);

	vector<string> recurrent_input_blob_names;
	RecurrentInputBlobNames(&recurrent_input_blob_names);

	// add the cell & states for padded area
	BlobShape recurrent_input_shape;
	recurrent_input_shape.add_dim(1);
	recurrent_input_shape.add_dim(this->num_);
	recurrent_input_shape.add_dim(num_output);

	for (int i = 0; i < recurrent_input_blob_names.size(); ++i) {
		net_param->add_input(recurrent_input_blob_names[i]);
		net_param->add_input_shape()->CopyFrom(recurrent_input_shape);
	}

	// concatenate the 4 hidden layers for 4 scanning directions
	LayerParameter h_concat_layer;
	h_concat_layer.set_type("Concat");
	h_concat_layer.set_name("concat_h");
	h_concat_layer.add_top("h");
	h_concat_layer.mutable_concat_param()->set_axis(1);

	for (int i = 1; i >= -1; i -= 2) {
		for (int j = 1; j >= -1; j -= 2) {
			string scan_dir = this->direction_to_str(i) + this->direction_to_str(j);
			// such as "x_{pp}"
			string x_input_blob_name = string("x_") + scan_dir;
			// W_{pp}_{xc}_x_{pp} = W^{pp}_{xc} * x^{pp} + b^{pp}_{c}
			LayerParameter* x_transform_param = net_param->add_layer();
			// such as "W_{pp}_{xc}_x_{pp}"
			string x_transform_param_top_name = string("W_") + scan_dir
					+ string("_xc_") + x_input_blob_name;
			x_transform_param->CopyFrom(biased_hidden_param);
			// such as "x_{pp}_{transform}"
			x_transform_param->set_name(x_input_blob_name + string("_transform"));
			// such as "W_{pp}_{xc}"
			x_transform_param->add_param()->set_name(
					string("W_") + scan_dir + string("_xc"));
			// such as "b_{pp}_c"
			x_transform_param->add_param()->set_name(
					string("b_") + scan_dir + string("_c"));

			x_transform_param->add_bottom(x_input_blob_name);
			x_transform_param->add_top(x_transform_param_top_name);

			LayerParameter* x_slice_param = net_param->add_layer();
			x_slice_param->CopyFrom(slice_param);
			x_slice_param->add_bottom(x_transform_param_top_name);
			x_slice_param->set_name((x_transform_param_top_name + string("_slice")));

			for (int py = 0; py < this->patch_ny_ + 1; ++py) {
				for (int px = 0; px < this->patch_nx_ + 1; ++px) {
					// such as "W_{pp}_{xc}_x_{pp}_0_0"
					x_slice_param->add_top(
							x_transform_param_top_name + string("_")
									+ this->coordinate_to_str(py, px));
				}
			}

			LayerParameter output_concat_layer;
			output_concat_layer.set_name(string("h_") + scan_dir + string("_concat"));
			output_concat_layer.set_type("Concat");
			// such as "h_{pp}"
			output_concat_layer.add_top(string("h_") + scan_dir);
			output_concat_layer.mutable_concat_param()->set_axis(2);

			int start_py = (i == 1 ? 1 : this->patch_ny_ - 1);
			int end_py = (i == 1 ? this->patch_ny_ : 0);
			int step_py = (i == 1 ? 1 : -1);
			int start_px = (j == 1 ? 1 : this->patch_nx_ - 1);
			int end_px = (j == 1 ? this->patch_nx_ : 0);
			int step_px = (j == 1 ? 1 : -1);

			const int min_py = step_py > 0 ? start_py : end_py;
			const int max_py = step_py > 0 ? end_py : start_py;
			const int min_px = step_px > 0 ? start_px : end_px;
			const int max_px = step_px > 0 ? end_px : start_px;

			for (int py = start_py; py >= min_py && py <= max_py; py += step_py) {
				for (int px = start_px; px >= min_px && px <= max_px; px += step_px) {
					if (step_py == 1) {
						CHECK_GE(py, start_py);
						CHECK_LE(py, end_py);
					} else {
						CHECK_GE(py, end_py);
						CHECK_LE(py, start_py);
					}
					if (step_px == 1) {
						CHECK_GE(px, start_px);
						CHECK_LE(px, end_px);
					} else {
						CHECK_GE(px, end_px);
						CHECK_LE(px, start_px);
					}

					string cur_yx = this->coordinate_to_str(py, px);
					string x_prev = this->coordinate_to_str(py, px - step_px);
					string y_prev = this->coordinate_to_str(py - step_py, px);

					// compute W^{pp}_{hxc} * h^{pp}_y_{x-1}
					string transform_xdir_top_name;
					{
						LayerParameter* transform_xdir = net_param->add_layer();
						transform_xdir->CopyFrom(hidden_param);
						// such as "transform_{pp}_xdir_y_x"
						transform_xdir->set_name(
								(string("transform_") + scan_dir + string("_xdir_") + cur_yx));
						// such as "W_{pp}_hxc"
						string param_name = string("W_") + scan_dir + string("_hxc");
						transform_xdir->add_param()->set_name(param_name);
						// such as "h_{pp}_y_{x-1}"
						string input_name = string("h_") + scan_dir + string("_") + x_prev;
//						transform_xdir->mutable_inner_product_param()->set_axis(2);
						transform_xdir->add_bottom(input_name);
						transform_xdir_top_name = param_name + string("_") + input_name;
						transform_xdir->add_top(transform_xdir_top_name);
					}

					// compute W^{pp}_{hyc} * h^{pp}_y_x
					string transform_ydir_top_name;
					{
						LayerParameter* transform_ydir = net_param->add_layer();
						transform_ydir->CopyFrom(hidden_param);
						// such as "transform_{pp}_ydir_y_x"
						transform_ydir->set_name(
								(string("transform_") + scan_dir + string("_ydir_") + cur_yx));
						// such as "W_{pp}_hyc"
						string param_name = string("W_") + scan_dir + string("_hyc");
						transform_ydir->add_param()->set_name(param_name);
						// such as "h_{pp}_{y-1}_x"
						string input_name = string("h_") + scan_dir + string("_") + y_prev;
						transform_ydir->add_bottom(input_name);
						transform_ydir_top_name = param_name + string("_") + input_name;
						transform_ydir->add_top(transform_ydir_top_name);
					}
					// Add the outputs of the linear transformations to compute the gate input.
					// gate^{pp}_input_y_x = W^{pp}_{hxc} * h^{pp}_y_x +  W^{pp}_{hyc} * h^{pp}_y_x + W^{pp}_{xc} * x^{pp} + b^{pp}_{xc}
					// gate^{pp}_input_y_x = [g'^{pp}_{ij}^t i'^{pp}_{ij}^t o'^{pp}_{ij}^t f'^{pp}_{ij}_x^t f'^{pp}_{ij}_y^t]

					// such as "gate_pp_input_y_x"
					string gate_input_top_name = string("gate_") + scan_dir
							+ string("_input_") + cur_yx;
					{
						LayerParameter* input_sum_layer = net_param->add_layer();
						input_sum_layer->set_type("Eltwise");
						input_sum_layer->mutable_eltwise_param()->set_operation(
								EltwiseParameter_EltwiseOp_SUM);
						input_sum_layer->set_name(
								(string("gate_") + scan_dir + string("_input_") + cur_yx));
						input_sum_layer->add_bottom(
								x_transform_param_top_name + string("_")
										+ this->coordinate_to_str(py, px));
						input_sum_layer->add_bottom(transform_xdir_top_name);
						input_sum_layer->add_bottom(transform_ydir_top_name);
						input_sum_layer->add_top(gate_input_top_name);
					}

					// Add LSTM2DUnitLayer to compute the cell & hidden vectors (e.g. C^{pp}_i_j, h^{pp}_i_j)
					// g^{pp}_{ij} = tanh(g'^{pp}_{ij})
					// i^{pp}_{ij} = sigmoid(i'^{pp}_{ij})
					// o^{pp}_{ij} =sigmoid(o'^{pp}_{ij})
					// f^{pp}_{ij}_x = sigmoid(f'^{pp}_{ij}_x)
					// f^{pp}_{ij}_y = sigmoid(f'^{pp}_{ij}_y)
					// c^{pp}_{ij} = f^{pp}_{ij}_x * c^{pp}_{i,j-1} + f^{pp}_{ij}_y * c^{pp}_{i-1,j} + i^{pp}_{ij} * g^{pp}_{ij}
					// h^{pp}_{ij} = o^{pp}_{ij} * tanh(c^{pp}_{ij})
					string lstm_2d_unit_layer_output_h_name = string("h_") + scan_dir
							+ string("_") + cur_yx;
					{
						LayerParameter* lstm_2d_unit_layer = net_param->add_layer();
						lstm_2d_unit_layer->set_type("LSTM2DUnit");
						// such as "lstm2d_unit_{pp}_y_x"
						lstm_2d_unit_layer->set_name(
								string("lstm2d_unit_") + scan_dir + string("_") + cur_yx);
						lstm_2d_unit_layer->mutable_lstm_2d_unit_param()->set_input_activation(
								input_activation_func_);
						lstm_2d_unit_layer->mutable_lstm_2d_unit_param()->set_output_activation(
								output_activation_func_);
						// such as "c_{pp}_y_{x-1}"
						lstm_2d_unit_layer->add_bottom(
								(string("c_") + scan_dir + string("_") + x_prev));
						// such as "c_{pp}_{y-1}_x"
						lstm_2d_unit_layer->add_bottom(
								(string("c_") + scan_dir + string("_") + y_prev));
						lstm_2d_unit_layer->add_bottom(gate_input_top_name);
						// such as "c_{pp}_y_x"
						lstm_2d_unit_layer->add_top(
								string("c_") + scan_dir + string("_") + cur_yx);
						lstm_2d_unit_layer->add_top(lstm_2d_unit_layer_output_h_name);
					}

					// Transpose h^{pp}_y_x of shape (1, N) into h^{pp}_y_x_t of shape (N, 1)
					// such as "h^{pp}_y_x_t"
					string reshaped_h_name = lstm_2d_unit_layer_output_h_name + string("_t");
					{
						LayerParameter* h_transpose_layer = net_param->add_layer();
						h_transpose_layer->set_type("Reshape");
						h_transpose_layer->set_name(
								string("reshape_h_") + scan_dir + string("_") + cur_yx);
						BlobShape* blobShape =
								h_transpose_layer->mutable_reshape_param()->mutable_shape();
						blobShape->Clear();
						blobShape->add_dim(this->num_);
						blobShape->add_dim(num_output);
						blobShape->add_dim(1);
						h_transpose_layer->add_bottom(lstm_2d_unit_layer_output_h_name);
						h_transpose_layer->add_top(reshaped_h_name);
					}
				} // for (int px = start_px; px > this->patch_nx_ || px < 0; px += step_px)
			} // for (int py = start_py; py > this->patch_ny_ || py < 0; py += step_py)

			start_py = (i == 1 ? 1 : 0);
			start_px = (j == 1 ? 1 : 0);
			for (int py = start_py; py < start_py + this->patch_ny_; py++) {
				for (int px = start_px; px < start_px + this->patch_nx_; px++) {
					string h_name = string("h_") + scan_dir + string("_")
							+ this->coordinate_to_str(py, px) + string("_t");
					output_concat_layer.add_bottom(h_name);
				}
			}
			net_param->add_layer()->CopyFrom(output_concat_layer);
			h_concat_layer.add_bottom(string("h_") + scan_dir);

		} // for (int j = 1; j >= -1; j -= 2)
	} // for (int i = 1; i >= -1; i -= 2)
	net_param->add_layer()->CopyFrom(h_concat_layer);
}

template<typename Dtype>
void LSTM2DLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
	names->resize(8 * (this->patch_nx_ + this->patch_ny_ + 1));
	int name_c = 0;
	for (int i = 1; i >= -1; i -= 2) {
		for (int j = 1; j >= -1; j -= 2) {
			int pad_row = i == 1 ? 0 : this->patch_ny_;
			int pad_col = j == 1 ? 0 : this->patch_nx_;
			for (int x = 0; x <= this->patch_nx_; ++x) {
				// such as "h_{pp}_0_1"
				(*names)[name_c++] = string("h_") + this->direction_to_str(i)
						+ this->direction_to_str(j) + string("_")
						+ this->coordinate_to_str(pad_row, x);
				// such as "c_{pp}_0_1"
				(*names)[name_c++] = string("c_") + this->direction_to_str(i)
						+ this->direction_to_str(j) + string("_")
						+ this->coordinate_to_str(pad_row, x);
			}
			int pad_col_start_y = i == 1 ? 1 : 0;
			int pad_col_end_y = pad_col_start_y + this->patch_ny_;
			for (int y = pad_col_start_y; y < pad_col_end_y; ++y) {
				// such as "h_{pp}_1_0"
				(*names)[name_c++] = string("h_") + this->direction_to_str(i)
						+ this->direction_to_str(j) + string("_")
						+ this->coordinate_to_str(y, pad_col);
				// such as "c_{pp}_1_0"
				(*names)[name_c++] = string("c_") + this->direction_to_str(i)
						+ this->direction_to_str(j) + string("_")
						+ this->coordinate_to_str(y, pad_col);
			}
		}
	}
	CHECK_EQ(name_c, 8 * (this->patch_nx_ + this->patch_ny_ + 1));
}

template<typename Dtype>
void LSTM2DLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
	names->resize(1);
	(*names)[0] = string("h");
}

INSTANTIATE_CLASS(LSTM2DLayer);
REGISTER_LAYER_CLASS(LSTM2D);

} // namespace caffe

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/lstm_2dlayer.hpp"

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
	return 2. * sigmoid(2. * x) - 1.;
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	num_output_ = this->layer_param_.lstm_2d_param().num_output();
	patch_h_ = this->layer_param_.lstm_2d_param().patch_height();
	patch_w_ = this->layer_param_.lstm_2d_param().patch_width();
	forget_gate_scaling_factor_ =
			this->layer_param_.lstm_2d_param().forget_gate_scaling_factor();

	CHECK_EQ(bottom[0]->num_axes(), 4);
	CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
	CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

	num_ = bottom[0]->shape(0);
	channels_ = bottom[0]->shape(1);
	patch_ny_ = bottom[0]->shape(2) / patch_h_;
	patch_nx_ = bottom[0]->shape(3) / patch_w_;
	int patch_size = channels_ * patch_h_ * patch_w_;

	X_.resize(4);
	H_.resize(4);
	C_.resize(4);
	T1_.resize(4);
	T2_.resize(4);
	T3_.resize(4);
	grad1_.resize(4);
	grad2_.resize(4);
	grad3_.resize(4);
	grad4_.resize(4);
	grad5_.resize(4);
	grad6_.resize(4);

	vector<int> X_shape(4);
	X_shape[0] = patch_ny_ + 1;
	X_shape[1] = patch_nx_ + 1;
	X_shape[2] = num_;
	X_shape[3] = patch_size;

	vector<int> cell_H_shape(4);
	cell_H_shape[0] = patch_ny_ + 1;
	cell_H_shape[1] = patch_nx_ + 1;
	cell_H_shape[2] = num_;
	cell_H_shape[3] = num_output_;

	vector<int> T1_shape(4);
	T1_shape[0] = patch_ny_ + 1;
	T1_shape[1] = patch_nx_ + 1;
	T1_shape[2] = num_;
	T1_shape[3] = num_output_ * 5;

	vector<int> T2_shape(4);
	T2_shape[0] = patch_ny_ + 1;
	T2_shape[1] = patch_nx_ + 1;
	T2_shape[2] = num_;
	T2_shape[3] = num_output_ * 2;

	vector<int> T3_shape(4);
	T3_shape[0] = patch_ny_ + 1;
	T3_shape[1] = patch_nx_ + 1;
	T3_shape[2] = num_;
	T3_shape[3] = num_output_ * 2;

	vector<int> T4_shape(4);
	T4_shape[0] = patch_ny_ + 1;
	T4_shape[1] = patch_nx_ + 1;
	T4_shape[2] = num_;
	T4_shape[3] = num_output_;

	vector<int> grad_shape(4);
	grad_shape[0] = patch_ny_ + 1;
	grad_shape[1] = patch_nx_ + 1;
	grad_shape[2] = num_;
	grad_shape[3] = num_output_;

	for (int i = 0; i < 4; ++i) {
		X_[i].reset(new Blob<Dtype>(X_shape));
		H_[i].reset(new Blob<Dtype>(cell_H_shape));
		C_[i].reset(new Blob<Dtype>(cell_H_shape));
		T1_[i].reset(new Blob<Dtype>(T1_shape));
		T2_[i].reset(new Blob<Dtype>(T2_shape));
		T3_[i].reset(new Blob<Dtype>(T3_shape));
		grad1_[i].reset(new Blob<Dtype>(grad_shape));
		grad2_[i].reset(new Blob<Dtype>(grad_shape));
		grad3_[i].reset(new Blob<Dtype>(grad_shape));
		grad4_[i].reset(new Blob<Dtype>(grad_shape));
		grad5_[i].reset(new Blob<Dtype>(grad_shape));
		grad6_[i].reset(new Blob<Dtype>(grad_shape));
	}

	num_blobs_per_dir_ = 10;
	this->blobs_.resize(4 * num_blobs_per_dir_);

	// W = [W_i;W_c;W_o;W_f]	shape: (num_output_*4, patch_size)
	vector<int> W_shape(2);
	W_shape[0] = num_output_ * 4;
	W_shape[1] = patch_size;
	// H^x = [H^x_i;H^x_c;H^x_o;H^x_f]	shape: (num_output_*4, num_output)
	vector<int> H_shape(2);
	H_shape[0] = num_output_ * 4;
	H_shape[1] = num_output_;
	// H^y = [H^y_i;H^y_c;H^y_o;H^y_f]	shape: (num_output_*4, num_output)
	// C^x = [C^x_i;C^x_f]	shape: (num_output_*2, num_output)
	vector<int> C_shape(2);
	C_shape[0] = num_output_ * 2;
	C_shape[1] = num_output_;
	// C^y = [C^y_i;C^y_f]	shape: (num_output_*2, num_output)
	// b_i shape: (num_output_)
	vector<int> b_shape(1, num_output_);
	// b_c shape: (num_output_)
	// b_o shape: (num_output_)
	// b^x_f shape: (num_output_)
	// b^y_f shape: (num_output_)

	shared_ptr<Filler<Dtype> > general_weight_filler(
			GetFiller<Dtype>(
					this->layer_param_.lstm_2d_param().general_weight_filler()));
	shared_ptr<Filler<Dtype> > general_bias_filler(
			GetFiller<Dtype>(
					this->layer_param_.lstm_2d_param().general_bias_filler()));
	shared_ptr<Filler<Dtype> > forget_gate_bias_filler(
			GetFiller<Dtype>(
					this->layer_param_.lstm_2d_param().forget_gate_bias_filler()));

	for (int dir_c = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, dir_c++) {
			this->blobs_[dir_c * num_blobs_per_dir_].reset(new Blob<Dtype>(W_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 1].reset(
					new Blob<Dtype>(H_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 2].reset(
					new Blob<Dtype>(H_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 3].reset(
					new Blob<Dtype>(C_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 4].reset(
					new Blob<Dtype>(C_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 5].reset(
					new Blob<Dtype>(b_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 6].reset(
					new Blob<Dtype>(b_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 7].reset(
					new Blob<Dtype>(b_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 8].reset(
					new Blob<Dtype>(b_shape));
			this->blobs_[dir_c * num_blobs_per_dir_ + 9].reset(
					new Blob<Dtype>(b_shape));

			for (int p = 0; p < 5; ++p) {
				general_weight_filler->Fill(
						this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
			}
			for (int p = 5; p < 8; ++p) {
				general_bias_filler->Fill(
						this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
			}
			for (int p = 8; p < 10; ++p) {
				forget_gate_bias_filler->Fill(
						this->blobs_[dir_c * num_blobs_per_dir_ + p].get());
			}
		}
	}

	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(4);
	top_shape[0] = num_;
	top_shape[1] = 4 * num_output_;
	top_shape[2] = patch_ny_;
	top_shape[3] = patch_nx_;
	top[0]->Reshape(top_shape);

	vector<int> bias_shape(1, num_);
	bias_multiplier_.Reshape(bias_shape);
	caffe_set(num_, Dtype(1), bias_multiplier_.mutable_cpu_data());
}

/*
 * implement the 2D LSTM model in paper '15 Scene Labeling with LSTM Recurrent Neural Networks'
 * */
template<typename Dtype>
void LSTM_2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	// fullfill X data
	for (int dir_c = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, dir_c++) {
			Dtype* X_data = X_[dir_c]->mutable_cpu_data();
			int X_data_idx = 0;
			for (int y = 0; y < patch_ny_ + 1; y++) {
				const int y2 = y - i;
				for (int x = 0; x < patch_nx_ + 1; x++) {
					const int x2 = x - j;
					if (y2 < 0 || y2 == patch_ny_ || x2 < 0 || x2 == patch_nx_) {
						for (int n = 0; n < num_; ++n) {
							for (int ch = 0; ch < channels_; ++ch) {
								for (int py = 0; py < patch_h_; py++) {
									for (int px = 0; px < patch_w_; px++) {
										X_data[X_data_idx++] = 0;
									}
								}
							}
						}
					} else {
						for (int n = 0; n < num_; ++n) {
							for (int ch = 0; ch < channels_; ++ch) {
								for (int py = 0; py < patch_h_; py++) {
									for (int px = 0; px < patch_w_; px++) {
										X_data[X_data_idx++] = bottom_data[bottom[0]->offset(n, ch,
												y2 * patch_h_ + py, x2 * patch_w_ + px)];
									}
								}
							}
						}
					}
				}
			}
		}
	}
	// reset cell state 'C' and hidden output 'H' of padded patches to zero
	for (int dir_c = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, dir_c++) {
			Dtype* H_data = H_[dir_c]->mutable_cpu_data();
			Dtype* C_data = C_[dir_c]->mutable_cpu_data();
			int pad_row = i == 1 ? 0 : patch_ny_;
			int pad_col = j == 1 ? 0 : patch_nx_;

			for (int x = 0; x < patch_nx_ + 1; ++x) {
				for (int n = 0; n < num_; ++n) {
					for (int d = 0; d < num_output_; ++d) {
						const int offset = H_[dir_c]->offset(pad_row, x, n, d);
						H_data[offset] = 0;
						C_data[offset] = 0;
					}
				}
			}

			for (int y = 0; y < patch_ny_ + 1; ++y) {
				for (int n = 0; n < num_; ++n) {
					for (int d = 0; d < num_output_; ++d) {
						const int offset = H_[dir_c]->offset(y, pad_col, n, d);
						H_data[offset] = 0;
						C_data[offset] = 0;
					}
				}
			}
		}
	}

	int patch_size = channels_ * patch_h_ * patch_w_;

	for (int dir_c = 0, i = 1; i >= 0; i--) {
		const int start_y = i == 1 ? 1 : patch_ny_ - 1;
		const int end_y = i == 1 ? patch_ny_ : 0;
		const int min_y = start_y <= end_y ? start_y : end_y;
		const int max_y = start_y <= end_y ? end_y : start_y;
		const int step_y = i == 1 ? 1 : -1;
		const int offset_y = i == 1 ? 1 : 0;

		for (int j = 1; j >= 0; j--, dir_c++) {
			const int start_x = j == 1 ? 1 : patch_nx_ - 1;
			const int end_x = j == 1 ? patch_nx_ : 0;
			const int min_x = start_x <= end_x ? start_x : end_x;
			const int max_x = start_x <= end_x ? end_x : start_x;
			const int step_x = j == 1 ? 1 : -1;
			const int offset_x = j == 1 ? 1 : 0;

			// W = [W_i;W_c;W_o;W_f]	shape: (num_output_*4, patch_size)
			const Dtype* param_W_data =
					this->blobs_[dir_c * num_blobs_per_dir_]->cpu_data();
			// H^x = [H^x_i;H^x_c;H^x_o;H^x_f]	shape: (num_output_*4, num_output)
			const Dtype* param_Hx_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 1]->cpu_data();
			// H^y = [H^y_i;H^y_c;H^y_o;H^y_f]	shape: (num_output_*4, num_output)
			const Dtype* param_Hy_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 2]->cpu_data();
			// C^x = [C^x_i;C^x_f]	shape: (num_output_*2, num_output)
			const Dtype* param_Cx_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 3]->cpu_data();
			// C^y = [C^y_i;C^y_f]	shape: (num_output_*2, num_output)
			const Dtype* param_Cy_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 4]->cpu_data();
			// b_i shape: (num_output_)
			const Dtype* bias_Bi_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 5]->cpu_data();
			// b_c shape: (num_output_)
			const Dtype* bias_Bc_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 6]->cpu_data();
			// b_o shape: (num_output_)
			const Dtype* bias_Bo_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 7]->cpu_data();
			// b^x_f shape: (num_output_)
			const Dtype* bias_Bxf_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 8]->cpu_data();
			// b^y_f shape: (num_output_)
			const Dtype* bias_Byf_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 9]->cpu_data();

			Dtype* X_data = X_[dir_c]->mutable_cpu_data();
			Dtype* H_data = H_[dir_c]->mutable_cpu_data();
			Dtype* C_data = C_[dir_c]->mutable_cpu_data();
			Dtype* T1_data = T1_[dir_c]->mutable_cpu_data();
			Dtype* T2_data = T2_[dir_c]->mutable_cpu_data();
			Dtype* T3_data = T3_[dir_c]->mutable_cpu_data();

			for (int y = start_y; y >= min_y && y <= max_y; y += step_y) {
				for (int x = start_x; x >= min_x && x <= max_x; x += step_x) {
					Dtype* T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					Dtype* T2_data_ptr = T2_data + T2_[dir_c]->offset(y, x);
					Dtype* T3_data_ptr = T3_data + T3_[dir_c]->offset(y, x);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, 4 * num_output_,
							patch_size, (Dtype) 1., X_data + X_[dir_c]->offset(y, x),
							param_W_data, (Dtype) 0., T1_data_ptr);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, 4 * num_output_,
							num_output_, (Dtype) 1.,
							H_data + H_[dir_c]->offset(y, x - step_x), param_Hx_data,
							(Dtype) 1., T1_data_ptr);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, 4 * num_output_,
							num_output_, (Dtype) 1.,
							H_data + H_[dir_c]->offset(y - step_y, x), param_Hy_data,
							(Dtype) 1., T1_data_ptr);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, 2 * num_output_,
							num_output_, (Dtype) 1.,
							C_data + C_[dir_c]->offset(y, x - step_x, 0, 0), param_Cx_data,
							(Dtype) 0., T2_data_ptr);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, 2 * num_output_,
							num_output_, (Dtype) 1.,
							C_data + C_[dir_c]->offset(y - step_y, x), param_Cy_data,
							(Dtype) 0., T3_data_ptr);

					// aggregate results in T1, T2 and T3 into T1

					// copy results of f^x_{y,x} into f^y_{y,x}
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							T1_data_ptr[4 * num_output_ + d] =
									T1_data_ptr[3 * num_output_ + d];
						}
						T1_data_ptr += (5 * num_output_);
					}

					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					T2_data_ptr = T2_data + T2_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							T1_data_ptr[d] += T2_data_ptr[d];
							T1_data_ptr[3 * num_output_ + d] = T2_data_ptr[num_output_ + d];
						}
						T1_data_ptr += (5 * num_output_);
						T2_data_ptr += (2 * num_output_);
					}

					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					T3_data_ptr = T3_data + T3_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							T1_data_ptr[d] += T3_data_ptr[d];
							T1_data_ptr[4 * num_output_ + d] = T3_data_ptr[num_output_ + d];
						}
						T1_data_ptr += (5 * num_output_);
						T3_data_ptr += (2 * num_output_);
					}

					// add bias terms
					// then apply activation functions to input gate, cell input and two forget gates
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							T1_data_ptr[d] = sigmoid(T1_data_ptr[d] + bias_Bi_data[d]);	// input gate i_{y,x}
							T1_data_ptr[num_output_ + d] = tanh(
									T1_data_ptr[num_output_ + d] + bias_Bc_data[d]);// cell input \hat{c}_{y,x}
							T1_data_ptr[2 * num_output_ + d] = sigmoid(
									T1_data_ptr[2 * num_output_ + d] + bias_Bo_data[d]); // output gate o_{y,x}
							T1_data_ptr[3 * num_output_ + d] = sigmoid(
									T1_data_ptr[3 * num_output_ + d] + bias_Bxf_data[d]); // x-direction forget gate f^x_{y,x}
							T1_data_ptr[4 * num_output_ + d] = sigmoid(
									T1_data_ptr[4 * num_output_ + d] + bias_Byf_data[d]); // y-direction forget gate f^y_{y,x}
							DLOG(INFO)<<"input gate value "<<T1_data_ptr[d]<<" cell input value "<<
							T1_data_ptr[num_output_ + d]<<" forget gate values "<<T1_data_ptr[3 * num_output_ + d]<<
							" "<<T1_data_ptr[4 * num_output_ + d];
						}
						T1_data_ptr += (5 * num_output_);
					}

							// compute cell state
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					Dtype* C_data_ptr = C_data + C_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							C_data_ptr[d] = T1_data_ptr[d] * T1_data_ptr[num_output_ + d]
									+ forget_gate_scaling_factor_
											* T1_data_ptr[3 * num_output_ + d]
											* C_data[C_[dir_c]->offset(y, x - step_x, n, d)]
									+ forget_gate_scaling_factor_
											* T1_data_ptr[4 * num_output_ + d]
											* C_data[C_[dir_c]->offset(y - step_y, x, n, d)];
							DLOG(INFO)<<"cell state value "<<C_data_ptr[d];
						}
						DLOG(INFO)<<"---";
						T1_data_ptr += (5 * num_output_);
						C_data_ptr += num_output_;
					}

							// compute cell hidden output h_{y,x}
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					C_data_ptr = C_data + C_[dir_c]->offset(y, x);
					Dtype* H_data_ptr = H_data + H_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							H_data_ptr[d] = T1_data_ptr[2 * num_output_ + d]
									* tanh(C_data_ptr[d]);
							DLOG(INFO)<<"cell hidden output "<<H_data_ptr[d];
						}
						DLOG(INFO)<<"---";
						T1_data_ptr += (5 * num_output_);
						C_data_ptr += num_output_;
						H_data_ptr += num_output_;
					}

							// copy cell hidden output h_{y,x} into top blob
					H_data_ptr = H_data + H_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							top_data[top[0]->offset(n, dir_c * num_output_ + d, y - offset_y,
									x - offset_x)] = H_data_ptr[d];
						}
						H_data_ptr += num_output_;
					}
				} // for (int x = start_x; x >= min_x && x <= max_x; x += step_x)
			} // for (int y = start_y; y >= min_y && y <= max_y; y += step_y)
		} // for (int j = 1; j >= 0; j--, dir_c++)
	} // for (int dir_c = 0, i = 1; i >= 0; i--)
}

template<typename Dtype>
void LSTM_2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// top shape (num, 4 * num_output_, patch_ny_, patch_nx_)
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bias_multiplier_data = bias_multiplier_.cpu_data();
	int patch_size = channels_ * patch_h_ * patch_w_;

	for (int dir_c = 0, i = 1; i >= 0; i--) {
		const int start_y = i == 1 ? 1 : patch_ny_ - 1;
		const int end_y = i == 1 ? patch_ny_ : 0;
		const int min_y = start_y <= end_y ? start_y : end_y;
		const int max_y = start_y <= end_y ? end_y : start_y;
		const int step_y = i == 1 ? 1 : -1;
		const int offset_y = i == 1 ? 1 : 0;

		for (int j = 1; j >= 0; j--, dir_c++) {
			const int start_x = j == 1 ? 1 : patch_nx_ - 1;
			const int end_x = j == 1 ? patch_nx_ : 0;
			const int min_x = start_x <= end_x ? start_x : end_x;
			const int max_x = start_x <= end_x ? end_x : start_x;
			const int step_x = j == 1 ? 1 : -1;
			const int offset_x = j == 1 ? 1 : 0;

			// W = [W_i;W_c;W_o;W_f]	shape: (num_output_*4, patch_size)
			Dtype* param_W_diff =
					this->blobs_[dir_c * num_blobs_per_dir_]->mutable_cpu_diff();
			const Dtype* param_W_data =
					this->blobs_[dir_c * num_blobs_per_dir_]->cpu_data();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_]->count() * sizeof(Dtype), 0,
					param_W_diff);
			// H^x = [H^x_i;H^x_c;H^x_o;H^x_f]	shape: (num_output_*4, num_output)
			Dtype* param_Hx_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 1]->mutable_cpu_diff();
			const Dtype* param_Hx_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 1]->cpu_data();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 1]->count() * sizeof(Dtype),
					0, param_Hx_diff);
			// H^y = [H^y_i;H^y_c;H^y_o;H^y_f]	shape: (num_output_*4, num_output)
			Dtype* param_Hy_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 2]->mutable_cpu_diff();
			const Dtype* param_Hy_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 2]->cpu_data();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 2]->count() * sizeof(Dtype),
					0, param_Hy_diff);
			// C^x = [C^x_i;C^x_f]	shape: (num_output_*2, num_output)
			Dtype* param_Cx_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 3]->mutable_cpu_diff();
			const Dtype* param_Cx_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 3]->cpu_data();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 3]->count() * sizeof(Dtype),
					0, param_Cx_diff);
			// C^y = [C^y_i;C^y_f]	shape: (num_output_*2, num_output)
			Dtype* param_Cy_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 4]->mutable_cpu_diff();
			const Dtype* param_Cy_data =
					this->blobs_[dir_c * num_blobs_per_dir_ + 4]->cpu_data();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 4]->count() * sizeof(Dtype),
					0, param_Cy_diff);
			// b_i shape: (num_output_)
			Dtype* bias_Bi_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 5]->mutable_cpu_diff();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 5]->count() * sizeof(Dtype),
					0, bias_Bi_diff);
			// b_c shape: (num_output_)
			Dtype* bias_Bc_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 6]->mutable_cpu_diff();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 6]->count() * sizeof(Dtype),
					0, bias_Bc_diff);
			// b_o shape: (num_output_)
			Dtype* bias_Bo_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 7]->mutable_cpu_diff();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 7]->count() * sizeof(Dtype),
					0, bias_Bo_diff);
			// b^x_f shape: (num_output_)
			Dtype* bias_Bxf_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 8]->mutable_cpu_diff();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 8]->count() * sizeof(Dtype),
					0, bias_Bxf_diff);
			// b^y_f shape: (num_output_)
			Dtype* bias_Byf_diff =
					this->blobs_[dir_c * num_blobs_per_dir_ + 9]->mutable_cpu_diff();
			caffe_memset(
					this->blobs_[dir_c * num_blobs_per_dir_ + 9]->count() * sizeof(Dtype),
					0, bias_Byf_diff);

			const Dtype* X_data = X_[dir_c]->cpu_data();
			const Dtype* H_data = H_[dir_c]->cpu_data();
			const Dtype* C_data = C_[dir_c]->cpu_data();
			const Dtype* T1_data = T1_[dir_c]->cpu_data();

			Dtype* X_diff = X_[dir_c]->mutable_cpu_diff();
			Dtype* H_diff = H_[dir_c]->mutable_cpu_diff();
			Dtype* C_diff = C_[dir_c]->mutable_cpu_diff();
			caffe_memset(X_[dir_c]->count() * sizeof(Dtype), 0, X_diff);
			caffe_memset(H_[dir_c]->count() * sizeof(Dtype), 0, H_diff);
			caffe_memset(C_[dir_c]->count() * sizeof(Dtype), 0, C_diff);

			Dtype* grad1_data = grad1_[dir_c]->mutable_cpu_data();
			Dtype* grad2_data = grad2_[dir_c]->mutable_cpu_data();
			Dtype* grad3_data = grad3_[dir_c]->mutable_cpu_data();
			Dtype* grad4_data = grad4_[dir_c]->mutable_cpu_data();
			Dtype* grad5_data = grad5_[dir_c]->mutable_cpu_data();
			Dtype* grad6_data = grad6_[dir_c]->mutable_cpu_data();

			for (int y = end_y; y >= min_y && y <= max_y; y -= step_y) {
				for (int x = end_x; x >= min_x && x <= max_x; x -= step_x) {
					// accumulate top diff into H_diff
					Dtype* H_diff_ptr = H_diff + H_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							H_diff_ptr[d] += top_diff[top[0]->offset(n,
									dir_c * num_output_ + d, y - offset_y, x - offset_x)];
						}
						H_diff_ptr += num_output_;
					}

					// compute intermediate and save into grad1 = d_h_{yx} * o_{yx} * f2'(c_{yx}) + d_c_{yx}
					H_diff_ptr = H_diff + H_[dir_c]->offset(y, x);
					Dtype* C_diff_ptr = C_diff + C_[dir_c]->offset(y, x);
					Dtype* grad1_data_ptr = grad1_data + grad1_[dir_c]->offset(y, x);
					const Dtype* T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					const Dtype* C_data_ptr = C_data + C_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							grad1_data_ptr[d] = H_diff_ptr[d]
									* T1_data_ptr[2 * num_output_ + d] * (1 - C_data_ptr[d] * C_data_ptr[d])
									+ C_diff_ptr[d];
						}
						H_diff_ptr += num_output_;
						C_diff_ptr += num_output_;
						grad1_data_ptr += num_output_;
						T1_data_ptr += (5 * num_output_);
						C_data_ptr += num_output_;
					}

					// compute intermediate gradient and save into grad2 = grad1 * \hat{c}_{yx} * f1'(\hat{i}_{yx})
					grad1_data_ptr = grad1_data + grad1_[dir_c]->offset(y, x);
					Dtype* grad2_data_ptr = grad2_data + grad2_[dir_c]->offset(y, x);
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							grad2_data_ptr[d] = grad1_data_ptr[d]
									* T1_data_ptr[num_output_ + d] * T1_data_ptr[d] * (1.0 - T1_data_ptr[d]);
						}
						grad1_data_ptr += num_output_;
						grad2_data_ptr += num_output_;
						T1_data_ptr += (5 * num_output_);
					}
					// compute gradient w.r.t. W_i
					const Dtype* X_data_ptr = X_data + X_[dir_c]->offset(y, x);
					grad2_data_ptr = grad2_data + grad2_[dir_c]->offset(y, x);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							patch_size, num_, (Dtype) 1., grad2_data_ptr, X_data_ptr,
							(Dtype) 1., param_Hx_diff);
					// compute gradient w.r.t. H^x_i
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad2_data_ptr,
							H_data + H_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Hx_diff);
					// compute gradient w.r.t H^y_i
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad2_data_ptr,
							H_data + H_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Hy_diff);
					// compute gradient w.r.t. C^x_i
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad2_data_ptr,
							C_data + C_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Cx_diff);
					// compute gradient w.r.t. C^y_i
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad2_data_ptr,
							C_data + C_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Cx_diff);
					// compute gradient w.r.t. b_i
					caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
							grad2_data_ptr, bias_multiplier_data, (Dtype) 1., bias_Bi_diff);

					// compute intermediate gradient and save into grad3 = grad1 * i_{y,x} * f2'(\hat{\hat{c_{y,x}}})
					grad1_data_ptr = grad1_data + grad1_[dir_c]->offset(y, x);
					Dtype* grad3_data_ptr = grad3_data + grad3_[dir_c]->offset(y, x);
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							const Dtype hat_c_yx = T1_data_ptr[num_output_ + d];
							grad3_data_ptr[d] = grad1_data_ptr[d] * T1_data_ptr[d]
									* (1 - hat_c_yx * hat_c_yx);
						}
						grad1_data_ptr += num_output_;
						grad3_data_ptr += num_output_;
						T1_data_ptr += (5 * num_output_);
					}
					// compute gradient w.r.t. W_c
					grad3_data_ptr = grad3_data + grad3_[dir_c]->offset(y, x);
					X_data_ptr = X_data + X_[dir_c]->offset(y, x);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							patch_size, num_, (Dtype) 1., grad3_data_ptr, X_data_ptr,
							(Dtype) 1., param_W_diff + num_output_ * patch_size);
					// compute gradient w.r.t H^x_c
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad3_data_ptr,
							H_data + H_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Hx_diff + num_output_ * num_output_);
					// compute gradient w.r.t H^y_c
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad3_data_ptr,
							H_data + H_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Hy_diff + num_output_ * num_output_);
					// compute gradient w.r.t. b_c
					caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
							grad3_data_ptr, bias_multiplier_data, (Dtype) 1., bias_Bc_diff);

					// compute intermediate gradient and save into grad4 = d_h_{y,x} * f2(c_{y,x}) * f1'(\hat{o_{y,x}})
					Dtype* grad4_data_ptr = grad4_data + grad4_[dir_c]->offset(y, x);
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					H_diff_ptr = H_diff + H_[dir_c]->offset(y, x);
					C_data_ptr = C_data + C_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							const Dtype o_yx = T1_data_ptr[2 * num_output_ + d];
							grad4_data_ptr[d] = H_diff_ptr[d] * tanh(C_data_ptr[d]) * o_yx
									* (1 - o_yx);
						}
						grad4_data_ptr += num_output_;
						T1_data_ptr += (5 * num_output_);
						H_diff_ptr += num_output_;
						C_data_ptr += num_output_;
					}
					// compute gradients w.r.t. W_o
					grad4_data_ptr = grad4_data + grad4_[dir_c]->offset(y, x);
					X_data_ptr = X_data + X_[dir_c]->offset(y, x);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							patch_size, num_, (Dtype) 1., grad4_data_ptr, X_data_ptr,
							(Dtype) 1., param_W_diff + 2 * num_output_ * patch_size);
					// compute gradients w.r.t. H^x_o
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad4_data_ptr,
							H_data + H_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Hx_diff + 2 * num_output_ * num_output_);
					// compute gradients w.r.t. H^y_o
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad4_data_ptr,
							H_data + H_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Hy_diff + 2 * num_output_ * num_output_);
					// compute gradients w.r.t. b_o
					caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
							grad4_data_ptr, bias_multiplier_data, (Dtype) 1., bias_Bo_diff);

					// compute intermediate gradient and save into
					// grad5 = forget_gate_scaling_factor_ * grad1 * C_{y,x-step_x} * f1'(\hat{f^x_{y,x}})
					// and grad6 = forget_gate_scaling_factor_ * grad1 * C_{y-step_y, x} * f1'(\hat{f^y_{y,x}})
					grad1_data_ptr = grad1_data + grad1_[dir_c]->offset(y, x);
					Dtype* grad5_data_ptr = grad5_data + grad5_[dir_c]->offset(y, x);
					Dtype* grad6_data_ptr = grad6_data + grad6_[dir_c]->offset(y, x);
					const Dtype* C_data_x_ptr = C_data + C_[dir_c]->offset(y, x - step_x);
					const Dtype* C_data_y_ptr = C_data + C_[dir_c]->offset(y - step_y, x);
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							const Dtype f_x_yx = T1_data_ptr[3 * num_output_ + d];
							const Dtype f_y_yx = T1_data_ptr[4 * num_output_ + d];
							grad5_data_ptr[d] = forget_gate_scaling_factor_ * grad1_data_ptr[d] * C_data_x_ptr[d]
									* f_x_yx * (1.0 - f_x_yx);
							grad6_data_ptr[d] = forget_gate_scaling_factor_ * grad1_data_ptr[d] * C_data_y_ptr[d]
									* f_y_yx * (1.0 - f_y_yx);
						}
						grad1_data_ptr += num_output_;
						grad5_data_ptr += num_output_;
						grad6_data_ptr += num_output_;
						C_data_x_ptr += num_output_;
						C_data_y_ptr += num_output_;
						T1_data_ptr += (5 * num_output_);
					}
					// compute gradients w.r.t. W_f
					grad5_data_ptr = grad5_data + grad5_[dir_c]->offset(y, x);
					grad6_data_ptr = grad6_data + grad6_[dir_c]->offset(y, x);
					X_data_ptr = X_data + X_[dir_c]->offset(y, x);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							patch_size, num_, (Dtype) 1., grad5_data_ptr, X_data_ptr,
							(Dtype) 1., param_W_diff + 3 * num_output_ * patch_size);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							patch_size, num_, (Dtype) 1., grad6_data_ptr, X_data_ptr,
							(Dtype) 1., param_W_diff + 3 * num_output_ * patch_size);

					// compute gradients w.r.t. H^x_f
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad5_data_ptr,
							H_data + H_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Hx_diff + 3 * num_output_ * num_output_);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad6_data_ptr,
							H_data + H_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Hx_diff + 3 * num_output_ * num_output_);
					// compute gradients w.r.t. H^y_f
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad5_data_ptr,
							H_data + H_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Hy_diff + 3 * num_output_ * num_output_);
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad6_data_ptr,
							H_data + H_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Hy_diff + 3 * num_output_ * num_output_);
					// compute gradients w.r.t. C^x_f
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad5_data_ptr,
							C_data + C_[dir_c]->offset(y, x - step_x), (Dtype) 1.,
							param_Cx_diff + num_output_ * num_output_);
					// compute gradients w.r.t. C^y_f
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
							num_output_, num_, (Dtype) 1., grad6_data_ptr,
							C_data + C_[dir_c]->offset(y - step_y, x), (Dtype) 1.,
							param_Cy_diff + num_output_ * num_output_);
					// compute gradients w.r.t b^x_f
					caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
							grad5_data_ptr, bias_multiplier_data, (Dtype) 1., bias_Bxf_diff);
					// compute gradients w.r.t b^y_f
					caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1.,
							grad6_data_ptr, bias_multiplier_data, (Dtype) 1., bias_Byf_diff);

					// update gradient w.r.t c_{y,x-step_x} and c_{y-step_y,x}
					T1_data_ptr = T1_data + T1_[dir_c]->offset(y, x);
					grad1_data_ptr = grad1_data + grad1_[dir_c]->offset(y, x);
					Dtype* C_x_diff_ptr = C_diff + C_[dir_c]->offset(y, x - step_x);
					Dtype* C_y_diff_ptr = C_diff + C_[dir_c]->offset(y - step_y, x);
					for (int n = 0; n < num_; ++n) {
						for (int d = 0; d < num_output_; ++d) {
							C_x_diff_ptr[d] += grad1_data_ptr[d]
									* T1_data_ptr[3 * num_output_ + d] * forget_gate_scaling_factor_;
							C_y_diff_ptr[d] += grad1_data_ptr[d]
									* T1_data_ptr[4 * num_output_ + d] * forget_gate_scaling_factor_;
						}
						T1_data_ptr += (5 * num_output_);
						grad1_data_ptr += num_output_;
						C_x_diff_ptr += num_output_;
						C_y_diff_ptr += num_output_;
					}

					grad2_data_ptr = grad2_data + grad2_[dir_c]->offset(y, x);
					grad5_data_ptr = grad5_data + grad5_[dir_c]->offset(y, x);
					grad6_data_ptr = grad6_data + grad6_[dir_c]->offset(y, x);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad2_data_ptr, param_Cx_data,
							(Dtype) 1., C_diff + C_[dir_c]->offset(y, x - step_x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad5_data_ptr,
							param_Cx_data + num_output_ * num_output_, (Dtype) 1.,
							C_diff + C_[dir_c]->offset(y, x - step_x));

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad2_data_ptr, param_Cy_data,
							(Dtype) 1., C_diff + C_[dir_c]->offset(y - step_y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad6_data_ptr,
							param_Cy_data + num_output_ * num_output_, (Dtype) 1.,
							C_diff + C_[dir_c]->offset(y - step_y, x));

					// update gradient w.r.t. H_{y,x-step_x} and H_{y-step_y,x}
					grad2_data_ptr = grad2_data + grad2_[dir_c]->offset(y, x);
					grad3_data_ptr = grad3_data + grad3_[dir_c]->offset(y, x);
					grad4_data_ptr = grad4_data + grad4_[dir_c]->offset(y, x);
					grad5_data_ptr = grad5_data + grad5_[dir_c]->offset(y, x);
					grad6_data_ptr = grad6_data + grad6_[dir_c]->offset(y, x);

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad2_data_ptr, param_Hx_data,
							(Dtype) 1., H_diff + H_[dir_c]->offset(y, x - step_x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad3_data_ptr,
							param_Hx_data + num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y, x - step_x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad4_data_ptr,
							param_Hx_data + 2 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y, x - step_x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad5_data_ptr,
							param_Hx_data + 3 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y, x - step_x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad6_data_ptr,
							param_Hx_data + 3 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y, x - step_x));

					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad2_data_ptr, param_Hy_data,
							(Dtype) 1., H_diff + H_[dir_c]->offset(y - step_y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad3_data_ptr,
							param_Hy_data + num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y - step_y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad4_data_ptr,
							param_Hy_data + 2 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y - step_y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad5_data_ptr,
							param_Hy_data + 3 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y - step_y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, num_output_,
							num_output_, (Dtype) 1., grad6_data_ptr,
							param_Hy_data + 3 * num_output_ * num_output_, (Dtype) 1.,
							H_diff + H_[dir_c]->offset(y - step_y, x));

					// compute gradients w.r.t. X
					grad2_data_ptr = grad2_data + grad2_[dir_c]->offset(y, x);
					grad3_data_ptr = grad3_data + grad3_[dir_c]->offset(y, x);
					grad4_data_ptr = grad4_data + grad4_[dir_c]->offset(y, x);
					grad5_data_ptr = grad5_data + grad5_[dir_c]->offset(y, x);
					grad6_data_ptr = grad6_data + grad6_[dir_c]->offset(y, x);
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, patch_size,
							num_output_, (Dtype) 1., grad2_data_ptr, param_W_data, (Dtype) 1.,
							X_diff + X_[dir_c]->offset(y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, patch_size,
							num_output_, (Dtype) 1., grad3_data_ptr,
							param_W_data + num_output_ * patch_size, (Dtype) 1.,
							X_diff + X_[dir_c]->offset(y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, patch_size,
							num_output_, (Dtype) 1., grad4_data_ptr,
							param_W_data + 2 * num_output_ * patch_size, (Dtype) 1.,
							X_diff + X_[dir_c]->offset(y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, patch_size,
							num_output_, (Dtype) 1., grad5_data_ptr,
							param_W_data + 3 * num_output_ * patch_size, (Dtype) 1.,
							X_diff + X_[dir_c]->offset(y, x));
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, patch_size,
							num_output_, (Dtype) 1., grad6_data_ptr,
							param_W_data + 3 * num_output_ * patch_size, (Dtype) 1.,
							X_diff + X_[dir_c]->offset(y, x));

				} // for (int x = end_x; x >= min_x && x <= max_x; x -= step_x)
			} // for (int y = end_y; y >= min_y && y <= max_y; y -= step_y)
		} // for (int j = 1; j >= 0; j--, dir_c++)
	} // for (int dir_c = 0, i = 1; i >= 0; i--)

	// copy gradients w.r.t X from X_ into bottom blob
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_memset(bottom[0]->count() * sizeof(Dtype), 0, bottom_diff);
	for (int dir_c = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, dir_c++) {
			const Dtype* X_diff = X_[dir_c]->cpu_diff();
			int X_diff_idx = 0;
			for (int y = 0; y < patch_ny_; ++y) {
				const int y2 = y - i;
				for (int x = 0; x < patch_nx_; ++x) {
					const int x2 = x - j;
					if (y2 < 0 || y2 == patch_ny_ || x2 < 0 || x2 == patch_nx_) {
						X_diff_idx += num_ * channels_ * patch_h_ * patch_w_;
					} else {
						for (int n = 0; n < num_; ++n) {
							for (int ch = 0; ch < channels_; ++ch) {
								for (int py = 0; py < patch_h_; ++py) {
									for (int px = 0; px < patch_w_; ++px) {
										bottom_diff[bottom[0]->offset(n, ch, y2 * patch_h_ + py,
												x2 * patch_w_ + px)] += X_diff[X_diff_idx++];
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(LSTM_2DLayer);
#endif

INSTANTIATE_CLASS(LSTM_2DLayer);
REGISTER_LAYER_CLASS(LSTM_2D);

} // namespace caffe

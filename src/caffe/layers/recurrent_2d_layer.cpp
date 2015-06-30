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
string Recurrent2DLayer<Dtype>::coordinate_to_str(const int r,
		const int c) const {
	ostringstream num;
	num << r << "_" << c;
	return num.str();
}

template<typename Dtype>
string Recurrent2DLayer<Dtype>::direction_to_str(const int i) const {
	CHECK_NE(i, 0);
	if (i > 0) {
		return string("p");
	} else {
		return string("n");
	}
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	patch_h_ = this->layer_param_.recurrent_2d_param().patch_height();
	patch_w_ = this->layer_param_.recurrent_2d_param().patch_width();
	LOG(INFO)<< "Recurrent2DLayer LayerSetUp patch_h_ " << patch_h_
	<< " patch_w_ " << patch_w_;
	// assume bottom[0] shape (num, channel, height, width)
	CHECK_EQ(bottom[0]->num_axes(), 4);
	CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0);
	CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0);

	num_ = bottom[0]->shape(0);
	channels_ = bottom[0]->shape(1);
	patch_ny_ = bottom[0]->shape(2) / patch_h_;
	patch_nx_ = bottom[0]->shape(3) / patch_w_;
	LOG(WARNING)<< "Recurrent2DLayer LayerSetUp patch_ny_ " << patch_ny_
	<< " patch_nx_ " << patch_nx_ << " num_ "<<num_;

	// Create a NetParameter; setup the inputs that aren't unique to particular
	// recurrent architectures.
	// As we fix the input image size,
	// The unrolled net structure is fixed as well
	NetParameter net_param;
	net_param.set_force_backward(true);

	BlobShape input_shape;
	input_shape.add_dim((patch_ny_ + 1) * (patch_nx_ + 1));
	input_shape.add_dim(num_);
	input_shape.add_dim(channels_);
	input_shape.add_dim(patch_h_);
	input_shape.add_dim(patch_w_);

	for (int p = 0, i = 1; i >= -1; i -= 2) {
		for (int j = 1; j >= -1; j -= 2, p++) {
			// four inputs: x_{pp}, x_{pn}, x_{np}, x_{nn}
			string input_name = string("x_") + direction_to_str(i)
					+ direction_to_str(j);
			net_param.add_input(input_name);
			net_param.add_input_shape()->CopyFrom(input_shape);
		}
	}

	this->FillUnrolledNet(&net_param);

	// Prepend this layer's name to the names of each layer in the unrolled net.
	const string& layer_name = this->layer_param_.name();
	if (layer_name.size() > 0) {
		for (int i = 0; i < net_param.layer_size(); ++i) {
			LayerParameter* layer = net_param.mutable_layer(i);
			layer->set_name(layer_name + "_" + layer->name());
		}
	}

	// Create the unrolled net.
	unrolled_net_.reset(new Net<Dtype>(net_param));
	unrolled_net_->set_debug_info(
			this->layer_param_.recurrent_2d_param().debug_info());

	x_input_blobs_.resize(4);
	for (int i = 1, p = 0; i >= -1; i -= 2) {
		for (int j = 1; j >= -1; j -= 2, p++) {
			// such as "x_{pp}","{x_pn}"
			string blob_name = string("x_") + direction_to_str(i)
					+ direction_to_str(j);
			x_input_blobs_[p] = CHECK_NOTNULL(
					unrolled_net_->blob_by_name(blob_name).get());
		}
	}

			// Setup pointers to outputs.
	vector<string> output_names;
	OutputBlobNames(&output_names);
	CHECK_EQ(output_names.size(), 1)<< "Output a single stack of 4 hidden layers for 4 scanning directions";
	output_blobs_.resize(output_names.size());
	for (int i = 0; i < output_names.size(); ++i) {
		output_blobs_[i] = CHECK_NOTNULL(
				unrolled_net_->blob_by_name(output_names[i]).get());
	}
		// This layer's parameters are any parameters in the layers of the unrolled
		// net. We only want one copy of each parameter, so check that the parameter
		// is "owned" by the layer, rather than shared with another.
	this->blobs_.clear();
	for (int i = 0; i < unrolled_net_->params().size(); ++i) {
		if (unrolled_net_->param_owners()[i] == -1) {
			DLOG(WARNING)<< "Adding parameter " << i << ": "
			<< unrolled_net_->param_display_names()[i];
			this->blobs_.push_back(unrolled_net_->params()[i]);
		}
	}

			// Check that param_propagate_down is set for all of the parameters in the
			// unrolled net; set param_propagate_down to true in this layer.
	for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
		for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
			CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
																																		<< "param_propagate_down not set for layer "
																																		<< i
																																		<< ", param "
																																		<< j;
		}
	}

	this->param_propagate_down_.clear();
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// In fact, there is only a single output blob
	CHECK_EQ(this->patch_ny_ * this->patch_nx_, output_blobs_[0]->shape(2));
	vector<int> topShape(4);
	topShape[0] = num_;
	topShape[1] = output_blobs_[0]->shape(1);
	topShape[2] = this->patch_ny_;
	topShape[3] = this->patch_nx_;
	// implicitly reshape [n, 4 * hidden_dim_, patch_ny_*patch_ny_]
	// into [n, 4*hidden_dim_, patch_ny_, patch_ny_]
	top[0]->Reshape(topShape);
	output_blobs_[0]->ShareData(*top[0]);
	output_blobs_[0]->ShareDiff(*top[0]);
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	prepare_patch_with_padding_cpu(bottom[0]);

	vector<string> recurrent_input_blob_names;
	this->RecurrentInputBlobNames(&recurrent_input_blob_names);
	// reset the hidden output (and cell state in the case of 2D LSTM) of padded patches to zero
	for (int i = 0; i < recurrent_input_blob_names.size(); ++i) {
		Blob<Dtype>* blob = unrolled_net_->blob_by_name(
				recurrent_input_blob_names[i]).get();
		Dtype* blob_data = blob->mutable_cpu_data();
		caffe_memset(blob->count() * sizeof(Dtype), 0, blob_data);
	}
	unrolled_net_->ForwardPrefilled();
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	unrolled_net_->Backward();
	back_propagate_grad_to_bottom_cpu(bottom[0]);
}

//template<typename Dtype>
//void Recurrent2DLayer<Dtype>::pad_image(const Blob<Dtype>* image) {
//	vector<int> pad_img_shape(3);
//	pad_img_shape[0] = image->shape(0);
//	pad_img_shape[1] = image->shape(1) + patch_h_;
//	pad_img_shape[2] = image->shape(2) + patch_w_;
//	const Dtype* image_data = image->cpu_data();
//
//	for (int p = 0, i = 1; i >= 0; i--) {
//		for (int j = 1; j >= 0; j--, p++) {
//			input_blobs_[p].reset(new Blob<Dtype>(pad_img_shape));
//			Dtype* data = input_blobs_[p]->mutable_cpu_data();
//			int start_x = j * patch_w_;
//			int start_y = i * patch_h_;
//			for (int c = 0; c < image->shape(0); ++c) {
//				int offset_image = image->offset(0, c);
//				int offset_input = input_blobs_[p]->offset(0, c, start_y, start_x);
//				for (int r = 0; r < image->shape(1); ++r) {
//					memcpy(data + offset_input, image_data + offset_image,
//							sizeof(Dtype) * image->shape(2));
//					offset_image += image->shape(2);
//					offset_input += pad_img_shape[2];
//				}
//			}
//		}
//	}
//}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::prepare_patch_with_padding_cpu(
		const Blob<Dtype>* image) {
	// assume image shape [n, ch, h, w]
	vector<int> input_blob_shape(5);
	input_blob_shape[0] = (patch_ny_ + 1) * (patch_nx_ + 1);
	input_blob_shape[1] = num_;
	input_blob_shape[2] = channels_;
	input_blob_shape[3] = patch_h_;
	input_blob_shape[4] = patch_w_;
	int patch_size = channels_ * patch_h_ * patch_w_;

	for (int p = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, p++) {
			x_input_blobs_[p]->Reshape(input_blob_shape);
			Dtype* data = x_input_blobs_[p]->mutable_cpu_data();
			for (int py = 0; py < patch_ny_ + 1; py++) {
				// y coordinate in original image
				int py2 = py - i;
				for (int px = 0; px < patch_nx_ + 1; px++) {
					// x coordinate in original image
					int px2 = px - j;
					for (int n = 0; n < num_; ++n) {
						if (px2 < 0 || py2 < 0 || px2 == patch_nx_ || py2 == patch_ny_) {
							for (int pix_c = 0, c = 0; c < channels_; ++c) {
								for (int y = 0; y < patch_h_; ++y) {
									for (int x = 0; x < patch_w_; ++x, pix_c++) {
										data[pix_c] = 0;
									}
								}
							}
						} else {
							for (int pix_c = 0, c = 0; c < channels_; ++c) {
								for (int y = 0; y < patch_h_; ++y) {
									const Dtype* image_data = image->cpu_data()
											+ image->offset(n, c, py2 * patch_h_ + y, px2 * patch_w_);
									for (int x = 0; x < patch_w_; ++x, pix_c++) {
										data[pix_c] = image_data[x];
									}
								}
							}
						}
						data += patch_size;
					}
				}
			}
		}
	}
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::back_propagate_grad_to_bottom_cpu(
		Blob<Dtype>* image) {
	Dtype* image_diff = image->mutable_cpu_diff();
	caffe_memset(image->count() * sizeof(Dtype), 0, image_diff);

	for (int p = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, p++) {
			const Dtype* x_input_blob_diff = x_input_blobs_[p]->cpu_diff();
			for (int py = 0; py < patch_ny_; py++) {
				// y coordinate in original image
				int py2 = py - i;
				for (int px = 0; px < patch_nx_; px++) {
					// x coordinate in original image
					int px2 = px - j;
					for (int n = 0; n < num_; ++n) {
						if (!(px2 < 0 || py2 < 0 || px2 == patch_nx_ || py2 == patch_ny_)) {
							for (int pix_c = 0, c = 0; c < channels_; ++c) {
								for (int y = 0; y < patch_h_; ++y) {
									Dtype* image_diff_local = image->mutable_cpu_diff()
											+ image->offset(n, c, py2 * patch_h_ + y, px2 * patch_w_);
									for (int x = 0; x < patch_w_; ++x, pix_c++) {
										image_diff_local[x] += x_input_blob_diff[pix_c];
									}
								}
							}
						}
						x_input_blob_diff += channels_ * patch_h_ * patch_w_;
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(Recurrent2DLayer, Forward);
#endif

INSTANTIATE_CLASS(Recurrent2DLayer);
} // namespace caffe

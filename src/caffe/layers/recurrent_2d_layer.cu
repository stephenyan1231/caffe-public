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
__global__ void prepare_patch_with_padding(const int nthreads,
		const Dtype* img_data, const int img_num, const int img_channels,
		const int img_height, const int img_width, const int patch_ny,
		const int patch_nx, const int patch_height, const int patch_width,
		const int y_offset, const int x_offset, Dtype* out_data) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int py = index / ((patch_nx + 1) * img_num);
		int rem = index % ((patch_nx + 1) * img_num);
		int px = rem / img_num;
		int n = rem % img_num;
		int py2 = py - y_offset;
		int px2 = px - x_offset;
		Dtype* out_data_ptr = out_data + ((py * (patch_nx + 1) + px) * img_num + n) * img_channels
				* patch_height * patch_width;
		if (px2 < 0 || py2 < 0 || px2 == patch_nx || py2 == patch_ny) {
			for (int pix_c = 0, c = 0; c < img_channels; ++c) {
				for (int y = 0; y < patch_height; ++y) {
					for (int x = 0; x < patch_width; ++x, ++pix_c) {
						out_data_ptr[pix_c] = 0;
					}
				}
			}
		} else {
			const Dtype* img_data_ptr = img_data + (n * img_channels * img_height * img_width);
			for (int pix_c = 0, c = 0; c < img_channels; ++c) {
				for (int y = 0; y < patch_height; ++y) {
					const Dtype* img_data_ptr2 = img_data_ptr
							+ (py2 * patch_height + y) * img_width + px2 * patch_width;
					for (int x = 0; x < patch_width; ++x, ++pix_c) {
						out_data_ptr[pix_c] = img_data_ptr2[x];
					}
				}
				img_data_ptr += (img_height * img_width);
			}
		}
	}
}

template<typename Dtype>
__global__ void back_propagate_grad_to_bottom(const int nthreads,
		Dtype* img_diff, const int img_num, const int img_channels,
		const int img_height, const int img_width, const int patch_ny,
		const int patch_nx, const int patch_height, const int patch_width,
		const int y_offset, const int x_offset, const Dtype* out_diff) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int py = index / ((patch_nx + 1) * img_num);
		int rem = index % ((patch_nx + 1) * img_num);
		int px = rem / img_num;
		int n = rem % img_num;
		int py2 = py - y_offset;
		int px2 = px - x_offset;
		if (!(px2 < 0 || py2 < 0 || px2 == patch_nx || py2 == patch_ny)) {
			const Dtype* out_diff_ptr = out_diff + ((py * (patch_nx + 1) + px) * img_num + n) * img_channels
					* patch_height * patch_width;
			Dtype* img_diff_ptr = img_diff + (n * img_channels * img_height * img_width);
			for (int pix_c = 0, c = 0; c < img_channels; ++c) {
				for (int y = 0; y < patch_height; ++y) {
					Dtype* img_diff_ptr2 = img_diff_ptr + (py2 * patch_height + y) * img_width
							+ px2 * patch_width;
					for (int x = 0; x < patch_width; ++x, ++pix_c) {
						img_diff_ptr2[x] += out_diff_ptr[pix_c];
					}
				}
				img_diff_ptr += (img_height * img_width);
			}
		}
	}
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	prepare_patch_with_padding_gpu(bottom[0]);

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
void Recurrent2DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	unrolled_net_->Backward();
	back_propagate_grad_to_bottom_gpu(bottom[0]);
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::prepare_patch_with_padding_gpu(
		const Blob<Dtype>* image) {
	// assume image shape [n, ch, h, w]
	vector<int> input_blob_shape(5);
	input_blob_shape[0] = (patch_ny_ + 1) * (patch_nx_ + 1);
	input_blob_shape[1] = num_;
	input_blob_shape[2] = channels_;
	input_blob_shape[3] = patch_h_;
	input_blob_shape[4] = patch_w_;

	const Dtype* image_data = image->gpu_data();
	const int img_num = image->shape(0);
	const int img_channels = image->shape(1);
	const int img_height = image->shape(2);
	const int img_width = image->shape(3);

	const int num_threads = (patch_ny_ + 1) * (patch_nx_ + 1) * num_;
	for (int p = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, p++) {
			x_input_blobs_[p]->Reshape(input_blob_shape);
			Dtype* data = x_input_blobs_[p]->mutable_gpu_data();
			prepare_patch_with_padding<Dtype> <<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
					num_threads, image_data, img_num, img_channels, img_height,
					img_width, patch_ny_, patch_nx_, patch_h_,patch_w_,i,j,data);
		}
	}
}

template<typename Dtype>
void Recurrent2DLayer<Dtype>::back_propagate_grad_to_bottom_gpu(
		Blob<Dtype>* image) {
	const int img_num = image->shape(0);
	const int img_channels = image->shape(1);
	const int img_height = image->shape(2);
	const int img_width = image->shape(3);

	Dtype* image_diff = image->mutable_cpu_diff();
	caffe_memset(image->count() * sizeof(Dtype), 0, image_diff);
	image_diff = image->mutable_gpu_diff();

	const int num_threads = (patch_ny_ + 1) * (patch_nx_ + 1) * num_;
	for (int p = 0, i = 1; i >= 0; i--) {
		for (int j = 1; j >= 0; j--, p++) {
			const Dtype* x_input_blob_diff = x_input_blobs_[p]->gpu_diff();
			back_propagate_grad_to_bottom<Dtype> <<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
					num_threads, image_diff, img_num, img_channels, img_height,
					img_width, patch_ny_, patch_nx_, patch_h_, patch_w_, i,j,x_input_blob_diff);
		}
	}
}

template void Recurrent2DLayer<float>::prepare_patch_with_padding_gpu(
		const Blob<float>* image);
template void Recurrent2DLayer<double>::prepare_patch_with_padding_gpu(
		const Blob<double>* image);

template void Recurrent2DLayer<float>::back_propagate_grad_to_bottom_gpu(
		Blob<float>* image);
template void Recurrent2DLayer<double>::back_propagate_grad_to_bottom_gpu(
		Blob<double>* image);

INSTANTIATE_LAYER_GPU_FUNCS(Recurrent2DLayer);

} // namespace caffe

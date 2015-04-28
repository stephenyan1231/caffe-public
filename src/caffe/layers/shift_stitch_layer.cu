#include "caffe/shift_stitch_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void ShiftStitchForward(const int nthreads, const Dtype *src_data,
		Dtype* tgt_data, int src_num, int channels, int src_height, int src_width,
		int stride_h, int stride_w, int tgt_num, int tgt_height, int tgt_width) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int w = index % src_width;
		const int h = (index / src_width) % src_height;
		const int ch = (index / src_width / src_height) % channels;
		const int sx = (index / src_width / src_height / channels) % stride_w;
		const int sy = (index / src_width / src_height / channels / stride_w)
				% stride_h;
		const int n = index / src_width / src_height / channels / stride_w
				/ stride_h;
		const int th = h * stride_h + sy;
		const int tw = w * stride_w + sx;
		int offset = ((n * channels + ch) * tgt_height + th) * tgt_width + tw;
		tgt_data[offset] = src_data[index];
	}
}

template<typename Dtype>
void ShiftStitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	nvtxMarkA("ShiftStitchLayer<Dtype>::Forward_gpu");
	int iter_out_num = num_;
	int iter_out_height = height_;
	int iter_out_width = width_;
	Blob<Dtype> *src_blob, *tgt_blob;
	const int count = top[0]->count();

	for (int i = 0; i < iter_; ++i) {
		iter_out_num /= (stride_h_[i] * stride_w_[i]);
		iter_out_height *= stride_h_[i];
		iter_out_width *= stride_w_[i];
		if (i == 0) {
			src_blob = bottom[0];
		} else {
			src_blob = tgt_blob;
		}
		if (i == (iter_ - 1)) {
			tgt_blob = top[0];
		} else {
			tgt_blob = new Blob<Dtype>(iter_out_num, channels_, iter_out_height,
					iter_out_width);
		}
		ShiftStitchForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, src_blob->gpu_data(), tgt_blob->mutable_gpu_data(),
				src_blob->num(), channels_, src_blob->height(), src_blob->width(),
				stride_h_[i], stride_w_[i], tgt_blob->num(), tgt_blob->height(),
				tgt_blob->width());
		if (i > 0) {
			delete src_blob;
		}
	}
//	LOG(INFO)<<"ShiftStitchLayer<Dtype>::Forward_gpu top shape "<<
//			top[0]->num()<<" "<<top[0]->channels()<<" "<<
//			top[0]->height()<<" "<<top[0]->width()<<" "<<
//			top[0]->count()*sizeof(Dtype);
	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
//		LOG(INFO)<<"shift stitch layer name "<<this->layer_param_.name();
//		size_t free_mem, total_mem;
//		cudaMemGetInfo(&free_mem, &total_mem);
//		LOG(INFO)<<"before: free memoey "<<free_mem<<" total_mem "<<total_mem;

		//		LOG(INFO)<<"shiftstitch layer name "<<this->layer_param_.name()<<
//				" free bottom "<<bottom[0]->count();
		bottom[0]->ReshapeForceMemoryFree(0, 0, 0, 0);

//		cudaMemGetInfo(&free_mem, &total_mem);
//		LOG(INFO)<<"after: free memoey "<<free_mem<<" total_mem "<<total_mem;

	}
}

template<typename Dtype>
__global__ void ShiftStitchBackward(const int nthreads, Dtype *src_diff,
		const Dtype* tgt_diff, int src_num, int channels, int src_height,
		int src_width, int stride_h, int stride_w, int tgt_num, int tgt_height,
		int tgt_width) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int w = index % src_width;
		const int h = (index / src_width) % src_height;
		const int ch = (index / src_width / src_height) % channels;
		const int sx = (index / src_width / src_height / channels) % stride_w;
		const int sy = (index / src_width / src_height / channels / stride_w)
				% stride_h;
		const int n = index / src_width / src_height / channels / stride_w
				/ stride_h;
		const int th = h * stride_h + sy;
		const int tw = w * stride_w + sx;
		int offset = ((n * channels + ch) * tgt_height + th) * tgt_width + tw;
		src_diff[offset] = tgt_diff[index];
	}
}

template<typename Dtype>
void ShiftStitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int iter_in_num = out_num_;
	int iter_in_height = out_height_;
	int iter_in_width = out_width_;
	const int count = bottom[0]->count();
	Blob<Dtype> *src_blob, *tgt_blob;
	for (int i = iter_ - 1; i >= 0; --i) {
		iter_in_num *= (stride_h_[i] * stride_w_[i]);
		iter_in_height /= stride_h_[i];
		iter_in_width /= stride_w_[i];
		if (i == (iter_ - 1)) {
			tgt_blob = top[0];
		} else {
			tgt_blob = src_blob;
		}
		if (i == 0) {
			src_blob = bottom[0];
		} else {
			src_blob = new Blob<Dtype>(iter_in_num, channels_, iter_in_height,
					iter_in_width);
		}
		ShiftStitchBackward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, src_blob->mutable_gpu_diff(),
				tgt_blob->gpu_diff(), src_blob->num(), channels_, src_blob->height(),
				src_blob->width(), stride_h_[i], stride_w_[i], tgt_blob->num(),
				tgt_blob->height(), tgt_blob->width());
		if (i < (iter_ - 1)) {
			delete tgt_blob;
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ShiftStitchLayer);
} // namespace caffe

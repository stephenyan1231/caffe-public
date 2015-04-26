#include "caffe/util/matrix_quantization.hpp"

namespace caffe {

//template<typename Dtype>
//__global__ void AssembleMatrix(const int nthreads, const Dtype* centers_data,
//		const unsigned short* indices_data, int num_center, int num_seg, int mat_height,
//		int mat_width, Dtype* mat_data) {
//	CUDA_KERNEL_LOOP(index, nthreads)
//	{
//		int seg_size = mat_width / num_seg;
//		int mat_y = index / num_seg;
//		int mat_x = index % num_seg;
//		int index = static_cast<int>(indices_data[mat_y * num_seg + mat_x]);
//		centers_data += (index * mat_width + mat_x * seg_size);
//		mat_data += (mat_y * mat_width + mat_x * seg_size);
//		for (int i = 0; i < seg_size; ++i) {
//			mat_data[i] = centers_data[i];
//		}
//	}
//}
//
//template __global__ void AssembleMatrix<float>(const int nthreads, const float* centers_data,
//		const unsigned short* indices_data, int num_center, int num_seg, int mat_height,
//		int mat_width, float* mat_data);
//template __global__ void AssembleMatrix<double>(const int nthreads, const double* centers_data,
//		const unsigned short* indices_data, int num_center, int num_seg, int mat_height,
//		int mat_width, double* mat_data);

template<typename Dtype>
__global__ void AssembleMatrix(const int nthreads, const Dtype* centers_data,
		const unsigned char* indices_data, int num_center, int num_seg, int mat_height,
		int mat_width, Dtype* mat_data) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int seg_size = mat_width / num_seg;
		int mat_y = index / num_seg;
		int mat_x = index % num_seg;
		int index = static_cast<int>(indices_data[mat_y * num_seg + mat_x]);
		centers_data += (index * mat_width + mat_x * seg_size);
		mat_data += (mat_y * mat_width + mat_x * seg_size);
		for (int i = 0; i < seg_size; ++i) {
			mat_data[i] = centers_data[i];
		}
	}
}

template __global__ void AssembleMatrix<float>(const int nthreads, const float* centers_data,
		const unsigned char* indices_data, int num_center, int num_seg, int mat_height,
		int mat_width, float* mat_data);
template __global__ void AssembleMatrix<double>(const int nthreads, const double* centers_data,
		const unsigned char* indices_data, int num_center, int num_seg, int mat_height,
		int mat_width, double* mat_data);

} // namespace caffe

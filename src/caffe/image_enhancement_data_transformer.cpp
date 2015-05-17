#include "caffe/image_enhancement_data_transformer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template<typename Dtype>
ImageEnhancementDataTransformer<Dtype>::ImageEnhancementDataTransformer(
		const ImageEnhancementTransformationParameter& param, Phase phase) :
		param_(param), phase_(phase) {
	const string& global_ftr_mean_file = param.global_ftr_mean_file();
	const string& semantic_context_ftr_mean_file =
			param.semantic_context_ftr_mean_file();
	const string& pixel_ftr_mean_file = param.pixel_ftr_mean_file();
	if (param.has_global_ftr_mean_file()) {
		LOG(INFO)<<"Loading global feature mean file:"<<global_ftr_mean_file;
		BlobProto global_ftr_mean_blob_proto;
		ReadProtoFromBinaryFileOrDie(global_ftr_mean_file.c_str(),
				&global_ftr_mean_blob_proto);
		global_ftr_mean_.FromProto(global_ftr_mean_blob_proto);
	}

	if (param.has_semantic_context_ftr_mean_file()) {
		LOG(INFO)<<"Loading semantic context feature mean file:"<<semantic_context_ftr_mean_file;
		BlobProto semantic_context_ftr_mean_blob_proto;
		ReadProtoFromBinaryFileOrDie(semantic_context_ftr_mean_file.c_str(),
				&semantic_context_ftr_mean_blob_proto);
		semantic_context_ftr_mean_.FromProto(semantic_context_ftr_mean_blob_proto);
	}

	if (param.has_pixel_ftr_mean_file()) {
		LOG(INFO)<<"Loading pixel feature mean file:"<<pixel_ftr_mean_file;
		BlobProto pixel_ftr_mean_blob_proto;
		ReadProtoFromBinaryFileOrDie(pixel_ftr_mean_file.c_str(),
				&pixel_ftr_mean_blob_proto);
		pixel_ftr_mean_.FromProto(pixel_ftr_mean_blob_proto);
	}
}

template<typename Dtype>
void ImageEnhancementDataTransformer<Dtype>::Transform(
		ImageEnhancementDatum& datum,
		shared_ptr<db::Transaction>& global_ftr_transaction,
		Blob<Dtype>* transformed_global_ftr,
		Blob<Dtype>* transformed_semantic_context_ftr,
		Blob<Dtype>* transformed_pixel_ftr) {
	Datum global_ftr_datum;
	global_ftr_datum.ParseFromString(
			global_ftr_transaction->GetValue(datum.image_name()));

	if (global_ftr_mean_.count() > 0) {
		CHECK_EQ(global_ftr_mean_.channels(), global_ftr_datum.channels());
	}
	const Dtype* global_ftr_mean_data = global_ftr_mean_.cpu_data();
	Dtype* transformed_global_ftr_data =
			transformed_global_ftr->mutable_cpu_data();
	for (int i = 0; i < global_ftr_mean_.channels(); ++i) {
		if (param_.has_global_ftr_mean_file()) {
			transformed_global_ftr_data[i] = (global_ftr_datum.float_data(i)
					- global_ftr_mean_data[i]) * param_.global_ftr_scale();
		} else {
			transformed_global_ftr_data[i] = global_ftr_datum.float_data(i)
					* param_.global_ftr_scale();
		}
	}

	CHECK_EQ(semantic_context_ftr_mean_.channels(),
			datum.semantic_context_ftr_size());
	const Dtype* semantic_context_ftr_mean_data =
			semantic_context_ftr_mean_.cpu_data();
	Dtype* transformed_semantic_context_ftr_data =
			transformed_semantic_context_ftr->mutable_cpu_data();
	for (int i = 0; i < semantic_context_ftr_mean_.channels(); ++i) {
		if (param_.has_semantic_context_ftr_mean_file()) {
			transformed_semantic_context_ftr_data[i] = (datum.semantic_context_ftr(i)
					- semantic_context_ftr_mean_data[i])
					* param_.semantic_context_ftr_scale();
		} else {
			transformed_semantic_context_ftr_data[i] = datum.semantic_context_ftr(i)
					* param_.semantic_context_ftr_scale();
		}
	}

	CHECK_EQ(pixel_ftr_mean_.channels(), datum.pixel_ftr_size());
	const Dtype* pixel_ftr_mean_data = pixel_ftr_mean_.cpu_data();
	Dtype* transformed_pixel_ftr_data = transformed_pixel_ftr->mutable_cpu_data();
	for (int i = 0; i < pixel_ftr_mean_.channels(); ++i) {
		if (param_.has_pixel_ftr_mean_file()) {
			transformed_pixel_ftr_data[i] = (datum.pixel_ftr(i)
					- pixel_ftr_mean_data[i]) * param_.pixel_ftr_scale();
		} else {
			transformed_pixel_ftr_data[i] = datum.pixel_ftr(i)
					* param_.pixel_ftr_scale();
		}

	}
}

INSTANTIATE_CLASS(ImageEnhancementDataTransformer);

} // namespace caffe

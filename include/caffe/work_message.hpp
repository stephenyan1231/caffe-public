#ifndef CAFFE_WORK_MESSAGE_HPP_
#define CAFFE_WORK_MESSAGE_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

enum MESSAGES {
	NO_WORK,
	COMPUTE_UPDATE_VALUE,
	UPDATE_WEIGHTS,
	FORWARD,
	FORWARD_PREFILLED,
	FORWARD_INPUT_BLOB_PROTOS,
	FORWARD_BACKWARD
};

class WorkMessage {
public:
	WorkMessage():message_type_(NO_WORK){}
	WorkMessage(MESSAGES message_type):message_type_(message_type){

	}
	~WorkMessage(){}

	MESSAGES getType(){
		return message_type_;
	}



protected:
	MESSAGES message_type_;

};

} // namespace caffe

#endif // CAFFE_WORK_MESSAGE_HPP_

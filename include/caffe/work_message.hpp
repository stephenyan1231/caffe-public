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
	FORWARD_FROM_TO,
	FORWARD_INPUT_BLOB_PROTOS,
	FORWARD_BACKWARD,
	BACKWARD,
	BACKWARD_FROM_TO,
	RESHAPE
};

class WorkMessage {
public:
	WorkMessage(){}
	~WorkMessage() {}

	virtual MESSAGES getType() = 0;

};

class NoWorkMessage: public WorkMessage{
public:
	NoWorkMessage():WorkMessage(){}
	virtual MESSAGES getType() {return NO_WORK;}
};

class ComputeUpdateValueMessage: public WorkMessage{
public:
	ComputeUpdateValueMessage():WorkMessage(){}
	virtual MESSAGES getType() {return COMPUTE_UPDATE_VALUE;}
};

class UpdateWeightsMessage: public WorkMessage{
public:
	UpdateWeightsMessage():WorkMessage(){}
	virtual MESSAGES getType() {return UPDATE_WEIGHTS;}
};

class ForwardMessage: public WorkMessage{
public:
	ForwardMessage():WorkMessage(){}
	virtual MESSAGES getType() {return FORWARD;}
};

class ForwardPrefilledMessage: public WorkMessage{
public:
	ForwardPrefilledMessage():WorkMessage(){}
	virtual MESSAGES getType() {return FORWARD_PREFILLED;}
};

class ForwardFromToMessage: public WorkMessage {
public:
	ForwardFromToMessage(int start, int end) :
			WorkMessage(), start_(start), end_(end) {
	}
	virtual MESSAGES getType() {return FORWARD_FROM_TO;}

	int get_start() {
		return start_;
	}
	int get_end() {
		return end_;
	}
protected:
	int start_, end_;
};

class ForwardInputBlobProtosMessage: public WorkMessage{
public:
	ForwardInputBlobProtosMessage():WorkMessage(){}
	virtual MESSAGES getType() {return FORWARD_INPUT_BLOB_PROTOS;}
};

class ForwardBackwardsMessage: public WorkMessage{
public:
	ForwardBackwardsMessage():WorkMessage(){}
	virtual MESSAGES getType() {return FORWARD_BACKWARD;}
};

class BackwardMessage: public WorkMessage{
public:
	BackwardMessage():WorkMessage(){}
	virtual MESSAGES getType() {return BACKWARD;}
};

class BackwardFromToMessage: public WorkMessage{
public:
	BackwardFromToMessage(int start, int end):WorkMessage(), start_(start), end_(end){}
	virtual MESSAGES getType() {return BACKWARD_FROM_TO;}
	int get_start() {
		return start_;
	}
	int get_end() {
		return end_;
	}
protected:
	int start_, end_;
};


class ReshapeMessage: public WorkMessage{
public:
	ReshapeMessage():WorkMessage(){}
	virtual MESSAGES getType() {return RESHAPE;}
};

} // namespace caffe

#endif // CAFFE_WORK_MESSAGE_HPP_

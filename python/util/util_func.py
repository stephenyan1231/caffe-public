import numpy as np
import sys
import os
import snappy
from joblib import Parallel, delayed

root_dir = os.environ['CAFFE_PROJ_DIR']
sys.path.append(os.path.join(root_dir, 'python/caffe/proto/'))
import caffe_pb2


def test_label(prob_dir, index, gt_label):
    datum = caffe_pb2.Datum()
    
    fn = '%010d' % index
    f = open(os.path.join(prob_dir, fn), 'rb')
    data = f.read()
    f.close()
    datum.ParseFromString(snappy.decompress(data))
    pred_lb = np.argmax(np.asarray(datum.float_data))
    return 1 if pred_lb == gt_label else 0


def read_layer_ftr(ftr_dir, index):
    fn = os.path.join(ftr_dir, '%010d' % index)
    try:
        f = open(fn, 'rb')
    except IOError:
        print 'can not open file %s' % fn
        return None
    datum = caffe_pb2.Datum()
    datum.ParseFromString(snappy.uncompress(f.read()))
    f.close()
    return np.asarray(datum.float_data, dtype=np.single)

def read_train_image_list(tr_img_list_file):
    # read a list of ordered training images
    f = open(tr_img_list_file, 'r')
    tr_img_names, tr_img_labels, tr_class_start = [], [], []
    cur_lb = -1
    for line in f:
        line = line.split(' ')
        lb = int(line[1])
        if not  lb == cur_lb:
            tr_class_start += [len(tr_img_names)]
            cur_lb = lb
        tr_img_names += [line[0]]
        tr_img_labels += [lb]   
    f.close()
     
    print '% d training images ' % len(tr_img_names)
    num_tr_imgs = len(tr_img_names)
    num_classes = len(tr_class_start)
    tr_class_end = tr_class_start[1:] + [num_tr_imgs]
    tr_class_start, tr_class_end = np.asarray(tr_class_start), np.asarray(tr_class_end)
    tr_class_size = tr_class_end - tr_class_start
    assert sum(tr_class_size) == num_tr_imgs
    print '%d classes' % num_classes
#     plt.hist(tr_class_size,bins=20)
#     plt.title('class size histogram')
    return tr_img_names, tr_img_labels, tr_class_start, tr_class_end

def write_train_image_list(tr_img_list_file, tr_img_names, tr_img_labels):
    assert(len(tr_img_names) == len(tr_img_labels))
    f=open(tr_img_list_file,'w')
    for i in range(len(tr_img_names)):
        f.write('%s %d' % (tr_img_names[i], tr_img_labels[i]))
    f.close()

def read_val_image_list(val_img_list_file):    
    # read a list of validation images
    f = open(val_img_list_file, 'r')
    val_img_names, val_img_labels = [], []
    for line in f:
        line = line.split(' ')
        lb = int(line[1])
        val_img_names += [line[0]]
        val_img_labels += [lb]    
    f.close()
    print '%d validation images ' % len(val_img_names)
    return val_img_names, val_img_labels

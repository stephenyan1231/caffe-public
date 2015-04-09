import numpy as np
import sys
import os
import snappy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cPickle
import gzip
import zipfile
from sklearn.cluster import KMeans
import multiprocessing as mtp
import time

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

def softmax(ftr):
    print 'compute softmax probabilities'
    num, dim = ftr.shape[0],ftr.shape[1]
    print 'num %d dim %d' % (num,dim)
    prob=np.zeros((num,dim))
    for i in range(num):
        max_val = np.max(ftr[i,:])
        row = ftr[i,:]-max_val
        exp_val=np.exp(row)
        prob[i,:]=exp_val / np.sum(exp_val)
    return prob    
    
''' compute a  histogram of the percentage of images 
that are classified as a label in the same cluster of its groundtruth label
'''
def plot_cluster_coverage(num_class, cluster_mbs,confusion_mat):
    cluster_coverage=np.zeros((num_class))
    for i in range(len(cluster_mbs)):
        for j in range(len(cluster_mbs[i])):
            class_id=cluster_mbs[i][j]
            cluster_coverage[class_id]=np.sum(confusion_mat[class_id,cluster_mbs[i]])
    n, bins, patches=plt.hist(cluster_coverage,bins=20)  
    

def read_text(fn):
    text=[]
    f=open(fn)
    for line in f:
        text+=[line]
    f.close()
    return text

def write_text(fn,text):
    f=open(fn,'w')
    f.writelines(text)
    f.close()
    
def top_k_accuracy(pred_labels, gt_labels, top_k):
    accu={}
    for k in top_k:
        assert pred_labels.shape[1]>=k
        count=0
        num=len(gt_labels)
        for i in range(num):
            for j in range(k):
                if pred_labels[i,j]==gt_labels[i]:
                    count+=1
                    break
        accu[str(k)]=float(count)/float(num)
    return accu

def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    if ms is not None and ms.file(filename).startswith('gzip'):
        fo = gzip.open(filename, 'rb')
        dict = cPickle.load(fo)
    elif ms is not None and ms.file(filename).startswith('Zip'):
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    
    fo.close()
    return dict

def find_layer_id(lay_names,lay_name):
    for i in range(len(lay_names)):
        if lay_names[i]==lay_name:
            return i
    return -1

def compute_compression_factor(m,n,n_kmean_cluster,n_seg):
    compress_factor= float(32*m*n)/ float(32*n_kmean_cluster*n+m*n_seg*32)
    print 'compress_factor %4.3f'%compress_factor    
    
def matrix_quantization(mat,n_kmean_cluster,n_seg):
    st_time=time.time()
    seg_size=mat.shape[1]/n_seg
    assert (mat.shape[1]%n_seg)==0
    mat_indices=np.zeros((mat.shape[0],n_seg),dtype=np.int32)
    mat_cluster_centers=np.zeros((n_kmean_cluster,mat.shape[1]), dtype=np.float32)
    kmeans=KMeans(init='k-means++',n_clusters=n_kmean_cluster,n_init=2,n_jobs=1,copy_x=False)
    
    display_num = 16
    for i in range(n_seg):
        start,end=i*seg_size,(i+1)*seg_size
        mat_seg=mat[:,start:end]
        mat_indices[:,i]=kmeans.fit_predict(mat_seg)
        if (i%(n_seg/display_num))==0:
            print mat_indices[:,i].shape, kmeans.cluster_centers_.shape
        mat_cluster_centers[:,start:end]=np.float32(kmeans.cluster_centers_)
    
    ep_time=time.time()-st_time
    print 'elapsed time %5.2f' % ep_time
        
    return mat_indices,mat_cluster_centers


def fine_layer_id(net_param, layer_name):
    for i in range(len(net_param.layer)):
        if layer_name == net_param.layer[i].name:
            return i
    print 'fail to find layer %s' % layer_name
    return -1


def quantiaztion_layer_parameter(args):
    st_time=time.time()
    
    net_param_fn, layer_name, layer_name_prefix, n_kmean_cluster, n_seg, save_dir=\
    args[0],args[1],args[2],args[3],args[4],args[5]
    print 'open net %s' % net_param_fn
    f=open(net_param_fn,'rb')
    net_param = caffe_pb2.NetParameter()
    net_param.ParseFromString(f.read())
    f.close()

    layer_id=fine_layer_id(net_param, layer_name_prefix+layer_name)
    
    fc_layer=net_param.layer[layer_id]
    print fc_layer.blobs[0].height,fc_layer.blobs[0].width      
    fc=np.asarray(fc_layer.blobs[0].data)
    fc=fc.reshape((fc_layer.blobs[0].height,fc_layer.blobs[0].width))
    
    compute_compression_factor(fc.shape[0],fc.shape[1],n_kmean_cluster[layer_name],n_seg[layer_name])

    fc_indices,fc_cluster_centers=matrix_quantization(fc,n_kmean_cluster[layer_name],n_seg[layer_name])

    blob_proto = caffe_pb2.BlobProto()
    blob_proto.num=1
    blob_proto.channels=1
    blob_proto.height=fc_cluster_centers.shape[0]
    blob_proto.width=fc_cluster_centers.shape[1]
    for j in range(fc_cluster_centers.shape[0]):
        for k in range(fc_cluster_centers.shape[1]):
            blob_proto.data.append(float(fc_cluster_centers[j,k]))
    fn=save_dir+layer_name_prefix+layer_name+'_%d_%d'%(n_kmean_cluster[layer_name],n_seg[layer_name])+\
    '_kmean_cluster_centers.binaryproto'
    print 'write to %s '%fn
    f=open(fn,"wb")
    f.write(blob_proto.SerializeToString())
    f.close()

    blob_proto = caffe_pb2.BlobProto()
    blob_proto.num=1
    blob_proto.channels=1
    blob_proto.height=fc_indices.shape[0]
    blob_proto.width=fc_indices.shape[1]
    for j in range(fc_indices.shape[0]):
        for k in range(fc_indices.shape[1]):
            blob_proto.data.append(float(fc_indices[j,k]))
    fn=save_dir+layer_name_prefix+layer_name+'_%d_%d'%(n_kmean_cluster[layer_name],n_seg[layer_name])+\
    '_kmean_cluster_indices.binaryproto'
    print 'write to %s' % fn
    f=open(fn,"wb")
    f.write(blob_proto.SerializeToString())
    f.close()
    
    ep_time=time.time()-st_time
    print 'elapsed time %5.2f' % ep_time
    
'''
read images.
preprocess using global contrast normalization and ZCA whitening
pack (image label, image pixel data) into binary files
data type: float
'''
import os
import sys
sys.path.append('/media/zyan3/label/proj/pylearn2/pylearn2/pylearn2')

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial


import numpy as np
import scipy.misc
import numpy.random

WIDTH,HEIGHT,CH= 32,32,3
gcn=55.


class CIFAR100_dataset(dense_design_matrix.DenseDesignMatrix):
    """
    The CIFAR-100 dataset.

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    gcn : WRITEME
    toronto_prepro : WRITEME
    axes : WRITEME
    start : WRITEME
    stop : WRITEME
    one_hot : WRITEME
    """

    def __init__(self, X, y,
            gcn = None,
            axes = ('b', 0, 1, 'c')):
        
        print X.shape,X.dtype

        assert X.max() == 255.
        assert X.min() == 0.
        X=np.cast['float32'](X)
        self.X=X
        self.y=y
        self.gcn = gcn
        if gcn is not None:
            assert isinstance(gcn,float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
            X *= gcn

        self.axes = axes
        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3),
                axes)

        super(CIFAR100_dataset,self).__init__(X=X, y=y, view_converter=view_converter)

        assert not np.any(np.isnan(self.X))
    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(rval[i,:]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example = False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the scale
        # determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i,:] /= np.abs(orig[i,:]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval
def global_contrast_normalization(X, gcn):
    assert isinstance(gcn,float)
    X = (X.T - X.mean(axis=1)).T
    X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
    X *= gcn
    return X
    
    
if __name__=='__main__':
    data_dir = '/media/zyan3/label/proj/caffe/caffe/data/cifar100/'
    tr_ori_dir = data_dir + 'train_imgs/'
    tr_aug_dir = data_dir + 'data_augmented/train_imgs/'
    ts_dir = data_dir + 'test_imgs'
    
    out_aug_batch_dir=data_dir + 'data_augmented/'
    if not os.path.exists(out_aug_batch_dir):
        os.mkdir(out_aug_batch_dir)

    class_ori_dirs=sorted(os.walk(tr_ori_dir).next()[1])
    print 'class_ori_dirs'
    print class_ori_dirs
    
    img_ori_n=0
    for class_dir in class_ori_dirs:
        print 'process class %s' % class_dir
        in_class_dir = os.path.join(tr_ori_dir,class_dir)
        in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
        print len(in_imgs)
        img_ori_n += len(in_imgs)
    print 'img_ori_n: %d' % img_ori_n
    '''load original training images'''
    imgs_ori_data=np.zeros((img_ori_n,WIDTH*HEIGHT*CH),dtype=np.uint8)
    imgs_ori_labels=np.zeros((img_ori_n),dtype=np.uint8)
    
    c=0
    for class_dir in class_ori_dirs:
        label = int(class_dir)
        print 'process class %s' % class_dir
        in_class_dir = os.path.join(tr_ori_dir,class_dir)
        in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
        for in_img in in_imgs:
            img_path=os.path.join(in_class_dir,in_img)
            img_data = scipy.misc.imread(img_path)
            imgs_ori_data[c,:]=img_data.flatten()
            imgs_ori_labels[c]=label
            c+=1
    print 'imgs_ori_data.shape'
    print imgs_ori_data.shape,imgs_ori_data.dtype
    print imgs_ori_data[1,:10]            
 
    train_ori=CIFAR100_dataset(imgs_ori_data,imgs_ori_labels,gcn=gcn)
    print 'train_ori shape'
    print train_ori.X.shape,train_ori.X.dtype               

    preprocessor = preprocessing.ZCA()
    train_ori.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
    print 'tag1.5,train_ori.X.shape,train_ori.X.dtype'
    print train_ori.X.shape,train_ori.X.dtype
    print train_ori.X[0,:10],train_ori.y[:10]    
   
    class_dirs=sorted(os.walk(tr_aug_dir).next()[1])
    print 'class_dirs'
    print class_dirs
 
     
    img_n = 0
    for class_dir in class_dirs:
#        print 'process class %s' % class_dir
        in_class_dir = os.path.join(tr_aug_dir,class_dir)
        in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
#        print len(in_imgs)
        img_n += len(in_imgs)
    print 'total image number: %d' % img_n
  
    #load training image data
    imgs_data=np.zeros((img_n,WIDTH*HEIGHT*CH),dtype=np.uint8)
    imgs_labels=np.zeros((img_n),dtype=np.uint8)
    c=0
    for class_dir in class_dirs:
        label = int(class_dir)
        print 'process class %s: label %d' % (class_dir, label)
        in_class_dir = os.path.join(tr_aug_dir,class_dir)
        in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
        for in_img in in_imgs:
            img_path=os.path.join(in_class_dir,in_img)
            img_data = scipy.misc.imread(img_path)
            imgs_data[c,:]=img_data.flatten()
            imgs_labels[c] = label
            c+=1
    print 'tag1,imgs_data.shape,imgs_data.dtype'
    print imgs_data.shape,imgs_data.dtype
    print imgs_data[0,:10],imgs_labels[:10]
     
    #randomly permuate training images
    train_permu=range(img_n)
    numpy.random.shuffle(train_permu)
    print 'train_permu'
    print train_permu[:10]
     
    imgs_data,imgs_labels=imgs_data[train_permu,:],imgs_labels[train_permu]
    print imgs_data[0,:10],imgs_labels[:10] 
 
    train_batch_num = 5
    train_batch_size = img_n / train_batch_num
 
    print 'train_batch_num,train_batch_size:%d %d' % (train_batch_num,train_batch_size)
    for i in range(train_batch_num):
        out_f = open(out_aug_batch_dir + 'float_data_batch_%d.bin' % (i+1),'wb')
        imgs_batch_data = imgs_data[i*train_batch_size:(i+1)*train_batch_size,:]
        imgs_batch_labels = imgs_labels[i*train_batch_size:(i+1)*train_batch_size]
        train_batch=CIFAR100_dataset(imgs_batch_data,imgs_batch_labels,gcn=gcn)
        train_batch.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
        print 'train_batch.X.shape,train_batch.X.dtype'
        print train_batch.X.shape,train_batch.X.dtype
        print train_batch.X[1,:10], train_batch.y[:10]
 
        for j in range(train_batch_size):
            out_f.write(chr(train_batch.y[j]))
            out_f.write(train_batch.X[j,:].tostring())
        out_f.close()
 
#     train=CIFAR100_dataset(imgs_data,imgs_labels,gcn=gcn)
#     print 'train.X.shape,train.X.dtype'
#     print train.X.shape,train.X.dtype
#    train.apply_preprocessor(preprocessor = preprocessor, can_fit = False)    
#    print 'tag2,train.X.shape,train.X.dtype'
#    print train.X.shape,train.X.dtype
#    print train.X[0,:10],train.y[:10]

# load testing image data
    test_imgs=sorted([img for img in os.listdir(ts_dir) if img.endswith('.jpg')])
    print 'test_imgs'
    print test_imgs[:10]
    
    test_labels_f = open(data_dir + 'test_labels.txt','r')
    test_labels=[]
    for line in test_labels_f:
        test_labels +=[int(line)]
    test_labels=np.asarray(test_labels)
    test_labels=np.cast['uint8'](test_labels)
    test_labels_f.close()
    
    test_num=len(test_imgs)
    print test_imgs[:10]
    print 'number of testing images:%d' % len(test_imgs)
    test_imgs_data=np.zeros((test_num,WIDTH*HEIGHT*CH),dtype=np.uint8)
    c=0
    for test_img in test_imgs:
        img_path=os.path.join(ts_dir,test_img)
        img_data=scipy.misc.imread(img_path)
        test_imgs_data[c,:]=img_data.flatten()
        c+=1
    print 'test_imgs_data shape'
    print test_imgs_data.shape,test_imgs_data.dtype
    test=CIFAR100_dataset(test_imgs_data,test_labels,gcn=gcn)
    test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
    
    print 'tag3,test.X.shape,test.X.dtype'
    print test.X.shape,test.X.dtype
    print test.X[0,:10],test.y[:10]
    
    out_f=open(out_aug_batch_dir + 'float_test_batch.bin','wb')
    for i in range(test_num):
        out_f.write(chr(test.y[i]))
        out_f.write(test.X[i,:].tostring())
    out_f.close()
    
    
    print 'hello'
    sys.exit()

'''Copyright 2014 Zhicheng Yan
Write preprocessed (global contrast normalization and ZCA whitening) cifar10 image data/labels into binary files 
'''

import cPickle
import numpy as np
import os

if __name__=='__main__':
    cifar10_data_batch_dir = r'/media/zyan3/drive1/Dropbox/Public/cifar10/cifar-10-gcn/'
    cifar10_out_data_batch_dir = r'/media/zyan3/drive1/Dropbox/private/proj/caffe_private_master/data/cifar10/'
    
    cifar100_data_batch_dir = r'/media/zyan3/label/proj/mavenlin-cuda-convnet/cuda-convnet/data/cifar100/cifar-100-whitened/'
    cifar100_out_dir_batch_dir = r'/media/zyan3/label/proj/caffe/caffe/data/cifar100/'    
    
    mnist_data_batch_dir = r'/media/zyan3/drive1/Dropbox/Public/mnist/mnist_normalized/'
    mnist_out_dir_batch_dir = r'/media/zyan3/drive1/Dropbox/private/proj/caffe_private_master/data/mnist/'    
    
#     for iter, item in enumerate([(cifar10_data_batch_dir,cifar10_out_data_batch_dir,5),\
#                                  (cifar100_data_batch_dir,cifar100_out_dir_batch_dir,5),\
#                                  (mnist_data_batch_dir,mnist_out_dir_batch_dir,6)]):
    for iter, item in enumerate([(cifar10_data_batch_dir,cifar10_out_data_batch_dir,5),(mnist_data_batch_dir,mnist_out_dir_batch_dir,6)]):
        data_batch_dir, out_data_batch_dir, train_batch_num = item[0], item[1], item[2]
        print 'data_batch_dir ',data_batch_dir
        print 'out_data_batch_dir ', out_data_batch_dir
        print 'train_batch_num:%d' % train_batch_num
        if not os.path.exists(out_data_batch_dir):
            os.mkdir(out_data_batch_dir)
        
        label_txt_tr = open(out_data_batch_dir + 'train_labels.txt','w')
        tr_c = 0
        for i in range(train_batch_num):
            data_batch_fn = data_batch_dir + 'data_batch_%d' % (i+1)
            data_label = cPickle.load(open(data_batch_fn))
            data, labels = data_label['data'], data_label['labels']
            print data.shape, len(labels)
            data = data.transpose()
            print data.dtype
            out_f=open(out_data_batch_dir + 'float_data_batch_%d.bin' % (i+1), 'wb')
            for j in range(data.shape[0]):
                out_f.write(chr(labels[j]))
                out_f.write(data[j,:].tostring())
                label_txt_tr.write('%d %d\n' % (tr_c,labels[j]))
                tr_c+=1
            out_f.close()            
        label_txt_tr.close()
          
        label_txt_ts = open(out_data_batch_dir + 'test_labels.txt','w')  
        data_batch_fn = data_batch_dir + 'data_batch_%d' % (train_batch_num+1)
        data_label = cPickle.load(open(data_batch_fn))
        data, labels = data_label['data'], data_label['labels']
        data = data.transpose()
        print data.dtype, data.shape
        out_f=open(out_data_batch_dir + 'float_test_batch.bin', 'wb')
        for j in range(data.shape[0]):
            out_f.write(chr(labels[j]))
            out_f.write(data[j,:].tostring())
            label_txt_ts.write('%d %d\n' % (j,labels[j]))
        out_f.close()
        label_txt_ts.close()
    
    print 'complete'
    
    
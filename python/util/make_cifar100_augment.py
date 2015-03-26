'''
read images.
pack (image label, image pixel data) into binary files
data type: uint8
'''
import os
import sys

import numpy as np
import scipy.misc
import numpy.random

WIDTH,HEIGHT,CH= 32,32,3
    
if __name__=='__main__':
    data_dir = '/media/zyan3/label/proj/caffe/caffe/data/cifar100/'
    tr_ori_dir = data_dir + 'train_imgs/'
    tr_aug_dir = data_dir + 'data_augmented_scale/train_imgs/'
    ts_dir = data_dir + 'test_imgs'
    
    out_aug_batch_dir=data_dir + 'data_augmented_scale/'
    if not os.path.exists(out_aug_batch_dir):
        os.mkdir(out_aug_batch_dir)

#     class_ori_dirs=sorted(os.walk(tr_ori_dir).next()[1])
#     print 'class_ori_dirs'
#     print class_ori_dirs
#     
#     img_ori_n=0
#     for class_dir in class_ori_dirs:
#         print 'process class %s' % class_dir
#         in_class_dir = os.path.join(tr_ori_dir,class_dir)
#         in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
#         print len(in_imgs)
#         img_ori_n += len(in_imgs)
#     print 'img_ori_n: %d' % img_ori_n
#     '''load original training images'''
#     imgs_ori_data=np.zeros((img_ori_n,WIDTH*HEIGHT*CH),dtype=np.uint8)
#     imgs_ori_labels=np.zeros((img_ori_n),dtype=np.uint8)
#     
#     c=0
#     for class_dir in class_ori_dirs:
#         label = int(class_dir)
#         print 'process class %s' % class_dir
#         in_class_dir = os.path.join(tr_ori_dir,class_dir)
#         in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]
#         for in_img in in_imgs:
#             img_path=os.path.join(in_class_dir,in_img)
#             img_data = scipy.misc.imread(img_path)
#             imgs_ori_data[c,:]=img_data.flatten()
#             imgs_ori_labels[c]=label
#             c+=1
#     print 'imgs_ori_data.shape'
#     print imgs_ori_data.shape,imgs_ori_data.dtype
#     print imgs_ori_data[1,:10]            
    
      
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
        out_f = open(out_aug_batch_dir + 'data_batch_%d.bin' % (i+1),'wb')
        imgs_batch_data = imgs_data[i*train_batch_size:(i+1)*train_batch_size,:]
        imgs_batch_labels = imgs_labels[i*train_batch_size:(i+1)*train_batch_size]
 
        for j in range(train_batch_size):
#             print 'label ',imgs_batch_labels[j]
            out_f.write(chr(imgs_batch_labels[j]))
            out_f.write(imgs_batch_data[j,:].tostring())
        out_f.close()
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
    
    out_f=open(out_aug_batch_dir + 'test_batch.bin','wb')
    for i in range(test_num):
        out_f.write(chr(test_labels[i]))
        out_f.write(test_imgs_data[i,:].tostring())
    out_f.close()
    
    
    print 'complete'
    sys.exit()

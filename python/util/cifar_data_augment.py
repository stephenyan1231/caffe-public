'''
data augmentation for alleviating overfitting in CNN training stage
Zhicheng Yan
Sep, 2014
'''

import sys
import os
from math import *
import numpy as np
import scipy.misc
import skimage.transform
import matplotlib.pyplot as plt


'''horizontal translation'''
def h_translate(im, deltax):
    h,w=im.shape[0],im.shape[1]
    im2=np.zeros_like(im)
    if deltax>=0:
        im2[:,deltax:w,:] = im[:,0:w-deltax,:]
        im2[:,:deltax,:] = im2[:,2*deltax-1:deltax-1:-1,:]
        
#         for i in range(0,deltax):
#             im2[:,i,:]=im2[:,deltax,:]
    else:
        im2[:,0:w+deltax,:]=im[:,-deltax:w,:]
        im2[:,w+deltax:w,:]=im2[:,w+deltax-1:w+2*deltax-1:-1,:]
#         for i in range(0,-deltax):
#             im2[:,w-1-i,:]=im2[:,w-1+deltax,:]
    return im2

'''vertical translation'''
def v_translate(im,deltay):
    h,w=im.shape[0],im.shape[1]
    im2=np.zeros_like(im)
    if deltay>=0:
        im2[deltay:h,:,:]=im[0:h-deltay,:,:]
        im2[:deltay,:,:]=im2[2*deltay-1:deltay-1:-1,:,:]
        
#         for i in range(0,deltay):
#             im2[i,:,:]=im2[deltay,:,:]
    else:
        im2[0:h+deltay,:,:]=im[-deltay:h,:,:]
        im2[h+deltay:h,:,:]=im2[h+deltay-1:h+2*deltay-1:-1,:,:]
#         
#         for i in range(0,-deltay):
#             im2[h-1-i,:,:]=im2[h-1+deltay,:,:]
        
    return im2

if __name__=='__main__':    
#     rot_degrees=range(-25,26,5)
#     print rot_degrees
    rot_degrees = []
    scales=np.linspace(1.0, 1.5, num=3)
    print scales
    x_translates,y_translates=[],[]
#     x_translates,y_translates=range(-12,13,4),range(-12,13,4)
#     x_translates.remove(0)
#     y_translates.remove(0)
    print x_translates,y_translates
    
   # rot_degrees=range(-25,26,25)
   # print rot_degrees
   # scales=np.linspace(1.1, 1.5, num=3)
   # print scales
   # x_translates,y_translates=range(-12,13,12),range(-12,13,12)
   # print x_translates,y_translates
    
         
    in_img_dir = '/media/zyan3/label/proj/caffe/caffe/data/cifar100/'
    out_img_dir = '/media/zyan3/label/proj/caffe/caffe/data/cifar100/data_augmented_scale/'
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
           
    in_tr_img_dir=in_img_dir + 'train_imgs/'
    out_tr_img_dir=out_img_dir + 'train_imgs/'
    if not os.path.exists(out_tr_img_dir):
        os.mkdir(out_tr_img_dir)
    
    class_dirs=sorted(os.walk(in_tr_img_dir).next()[1])
    print class_dirs
    
    for class_dir in class_dirs:
        print 'process class %s' % class_dir
        in_class_dir = in_tr_img_dir + class_dir
        out_class_dir = out_tr_img_dir + class_dir
        if not os.path.exists(out_class_dir):
            os.mkdir(out_class_dir)
        
        in_imgs =  [img for img in os.listdir(in_class_dir) if img.endswith('.jpg')]        
        
        
        print '[',
        for i in range(len(in_imgs)):
#         for i in range(2):    
            print '.',
            in_im=scipy.misc.imread(os.path.join(in_class_dir,in_imgs[i]))
            h,w=in_im.shape[0],in_im.shape[1]
            for j in range(len(rot_degrees)):
                rot_img = skimage.transform.rotate(in_im,rot_degrees[j],mode='reflect')
                save_img_fn = '%s_rot_%04d.jpg' % (in_imgs[i][:-4],rot_degrees[j])
                scipy.misc.imsave(os.path.join(out_class_dir,save_img_fn),rot_img)
            for j in range(len(scales)):
                scaled_img_raw = skimage.transform.rescale(in_im,scale=scales[j],mode='reflect')
                h2,w2=scaled_img_raw.shape[0],scaled_img_raw.shape[1]
                scaled_img=scaled_img_raw[floor(h2/2)-floor(h/2):floor(h2/2)+floor(h/2),floor(w2/2)-floor(w/2):floor(w2/2)+floor(w/2),:]
                assert scaled_img.shape == in_im.shape
                save_img_fn = '%s_scaled_%3.2f.jpg' % (in_imgs[i][:-4],scales[j])
                scipy.misc.imsave(os.path.join(out_class_dir,save_img_fn),scaled_img)
            for j in range(len(x_translates)):
                x_img = h_translate(in_im,x_translates[j])
                save_img_fn = '%s_x_translate_%03d.jpg' % (in_imgs[i][:-4],x_translates[j])
                scipy.misc.imsave(os.path.join(out_class_dir,save_img_fn),x_img)
            for j in range(len(y_translates)):
                y_img = v_translate(in_im,y_translates[j])
                save_img_fn = '%s_y_translate_%03d.jpg' % (in_imgs[i][:-4],y_translates[j])    
                scipy.misc.imsave(os.path.join(out_class_dir,save_img_fn),y_img)
        print ']'
#     for roots, dirs, files in os.walk(in_tr_img_dir):
#         print roots
#         for dir_nm in dirs:
#             print '\tdir:%s' % dir_nm
#               
#         for f_nm in files:
#             print '\tfile:%s' % f_nm
#     
    
#     im=scipy.misc.imread()
    print 'complete'
    sys.exit()
    
    

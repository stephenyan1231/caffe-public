'''Copyright (c) 2015 Zhicheng Yan (zhicheng.yan@live.com)
'''
import sys
import os
import argparse
import lmdb

sys.path.append(os.environ['CAFFE_PROJ_DIR'] + '../dl-image-enhance/py')
sys.path.append(os.environ['CAFFE_PROJ_DIR'] + 'python/caffe/proto')
import caffe_pb2

from utilCnnImageEnhance import *

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--image_list_file',help="image list file path")
    arg_parser.add_argument('--tiff_image_dir',help="folder where 16-bit tiff AdobeRGB color space images reside")
    arg_parser.add_argument('--save_LAB_image_lmdb',help="lmdb database path where LAB images are saved")
    args = arg_parser.parse_args()
    print 'image_list_file:', args.image_list_file    
    print 'tiff_image_dir:', args.tiff_image_dir    
    print 'save_LAB_image_lmdb:', args.save_LAB_image_lmdb    
    
    print 'read tiff 16-bit image into LAB space and export them into binary files'    
    dl_image_enhance_dir = os.environ['CAFFE_PROJ_DIR'] + '/../dl-image-enhance/'
    
    list_f=open(args.image_list_file    )
    img_nms=[]
    for line in list_f.readlines():
         img_nms += [line[:-1]]
    list_f.close()
    
    print '%d images' % len(img_nms)
    
    LAB_img_lmdb = lmdb.Environment(args.save_LAB_image_lmdb, map_size = 4294967295)
    LAB_img_lmdb_txn=LAB_img_lmdb.begin(write=True)
    commit_frequency = 10
    for i,img_nm in enumerate(img_nms):
        original_img = np.single(read_tiff_16bit_img_into_LAB(args.tiff_image_dir  + img_nm + '.tif'))
        print original_img.shape, original_img.dtype
        
        datum = caffe_pb2.Datum()
        datum.height = original_img.shape[0]
        datum.width = original_img.shape[1]
        datum.channels = original_img.shape[2]
        for c in range(datum.channels):
            for h in range(datum.height):
                for w in range(datum.width):
                    datum.float_data.append(float(original_img[h,w,c]))
        print 'datum float data length ',len(datum.float_data)
        
        LAB_img_lmdb_txn.put(img_nm,datum.SerializeToString())
        if (i+1) % commit_frequency == 0:
            LAB_img_lmdb_txn.commit()
            LAB_img_lmdb_txn=LAB_img_lmdb.begin(write=True)
            
    if not len(img_nms) % commit_frequency == 0:
        LAB_img_lmdb_txn.commit()
    LAB_img_lmdb.close() 
    
    '''sanity check. expect to have 115 keys'''
    print 'sanity check'
    LAB_img_lmdb = lmdb.open(args.save_LAB_image_lmdb)
    LAB_img_lmdb_txn=LAB_img_lmdb.begin(write=False)
    LAB_img_lmdb_cursor=LAB_img_lmdb_txn.cursor()
    count = 0
    # print '%d keys' % len(prob_cursor.keys())
    for key, value in LAB_img_lmdb_cursor:
        print key
        count += 1
    print '%d keys' % count
        
        
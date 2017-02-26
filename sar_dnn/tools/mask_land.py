#! /usr/bin/python
import os
import numpy as np
import cv2

def resize_prediction(predict_dir, hhv_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    date = os.listdir(predict_dir)
    date = [d[0:15] for d in date if d.find('.tif') != -1]
    for d in date:
        ia = cv2.imread(hhv_dir + '/' + d + '_IA.tif', -1)
        im = cv2.imread(predict_dir + '/' + d + '.tif', -1)
        im = cv2.resize(im, (ia.shape[1], ia.shape[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_dir + '/' + d + '.tif', im)

# make land pixels to 0
def mask_land(hhv_dir, mask_dir, out_dir):
    date = os.listdir(hhv_dir)
    date = [d for d in date if d.find('.tif') != -1]
    for d in date:
        im = cv2.imread(hhv_dir + '/' + d, -1)
        mask = cv2.imread(mask_dir + '/' + d[0:15] + '-mask.tif', -1)
        im[mask != 0] = 0
        cv2.imwrite(out_dir + '/' + d, im)



if __name__ == '__main__':
    predict_dir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_IA_45/predict_stage1'
    hhv_dir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/hhv'
    mask_dir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/mask_land_free'
    out_dir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/hhv_land_masked'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mask_land(hhv_dir, mask_dir, out_dir)
    #resize_prediction(predict_dir, hhv_dir, out_dir)

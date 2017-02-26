#! /usr/bin/python

import sys
import cPickle
import gzip
import os
import sys
import Image
import numpy as np
import math
import gdal
from gdalconst import *
import caffe
import cv2
import argparse
from scipy.interpolate import griddata

def export_proto(input, target, outpath):
    # construct batchs
    vecstr = caffe.io.arraylist_to_datumvector_str(input, target)
    # write batches
    f = open(outpath, "wb")
    f.write(vecstr)
    f.close()

def patch_img(image, points, window):
    #img is a image matrix
    #samples are the locations to extract patches
    #window is the patchsize, should be a odd number, if not, it will be transfered to a odd number by +1
    # check windows size
    if window%2 == 0:
        print "window has to be uneven. abord"
        return
    rl = int(math.floor(window/2))
    rr = int(rl)

    #select available points:
    #conditions: not masked as 0, in image, not out of boundary
    inputs = []
    target = []
    subpoints = []
    for i,point in zip(range(len(points)), points):
         if point[0] >= rl and point[0] < image.shape[2]-rr and \
            point[1] >= rl and point[1] < image.shape[1]-rr :
            inputs.append(image[:,int(point[1])-rl:int(point[1])+rr+1,
            int(point[0])-rl:int(point[0])+rr+1])
            target.append(int(point[2]))
            subpoints.append(point)
    return inputs, target,subpoints

def patch_img_label_multi_scale(image, scale, points, window):
    #img is a image matrix
    #samples are the locations to extract patches
    #window is the patchsize, should be a odd number, if not, it will be transfered to a odd number by +1
    # check windows size
    if window%2 == 0:
        print "window has to be uneven. abord"
        return
    rl = int(math.floor(window/2))
    rr = int(rl)
    bound = (rl+1) / scale
    #select available points:
    #conditions: not masked as 0, in image, not out of boundary
    input = []
    target = []
    subpoints = []
    image_global = [cv2.resize(i, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR) for i in image]
    image_global = np.asarray(image_global)
    for i,point in zip(range(len(points)), points):
         if point[0] >= bound and point[0] < image.shape[2]-bound and \
            point[1] >= bound and point[1] < image.shape[1]-bound :
            patch = image[:,int(point[1])-rl:int(point[1])+rr+1,
                int(point[0])-rl:int(point[0])+rr+1]
            patch2 = image_global[:, int(point[1] * scale) - rl:int(point[1]*scale)+rr+1,
                    int(point[0] * scale) - rl:int(point[0] * scale) + rr + 1 ]
            input.append(np.concatenate((patch, patch2), axis = 0 ) )
            target.append(int(point[2]))
            subpoints.append(point)
    return input, target, subpoints

def patch_img_label(image, label, points, window):
    #img is a image matrix
    #samples are the locations to extract patches
    #window is the patchsize, should be a odd number, if not, it will be transfered to a odd number by +1
    # check windows size
    if window%2 == 0:
        print "window has to be uneven. abord"
        return
    rl = int(math.floor(window/2))
    rr = int(rl)
    bound = rl
    #select available points:
    #conditions: not masked as 0, in image, not out of boundary
    input = []
    target = []
    subpoints = []
    for i,point in zip(range(len(points)), points):
         if point[0] >= bound and point[0] < image.shape[2]-bound and \
            point[1] >= bound and point[1] < image.shape[1]-bound :
            input.append(image[:,int(point[1])-rl:int(point[1])+rr+1,
                int(point[0])-rl:int(point[0])+rr+1])
            target.append(label[int(point[1])-rl:int(point[1])+rr+1,
                int(point[0])-rl:int(point[0])+rr+1])
            subpoints.append(point)
    return input, target,subpoints

def gen_sample_locs(label, sample_rate, ignore_value, margin):
    """
    generate samples using label image. More samples at the boundaries of the different classes, maybe

    label: the label nparray
    nsample: number of samples
    margin: min distance of sample locations to image margins
    """
    nsample = label.shape[0] * label.shape[1] * sample_rate
    rows = np.random.random_integers(margin, label.shape[0] - margin - 1, (nsample, 1))
    cols = np.random.random_integers(margin, label.shape[1] - margin - 1, (nsample, 1))
    locs = np.concatenate((cols, rows), axis = 1)
    locs = [l for l in locs if label[l[1], l[0]] != ignore_value ]
    return np.asarray(locs)


def read_image(filein):
    dataset = gdal.Open(filein, GA_ReadOnly )
    if dataset is None:
        print 'file does not exist:'
        print filein
        raise Exception("file doesnot exist")
        return
    return dataset.ReadAsArray()

def gen_fcn_sample(window, sardir, image_subfixes, labeldir, label_subfix, sample_rate, ignore_value, output_dir):
    """
        genenrate ice water samples from labeled images
    """
    labellist = open(labeldir + '/source.txt').readlines()
    labellist = [l.strip() for l in labellist]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_used_dir = output_dir + '/label_used'
    if not os.path.exists(label_used_dir):
        os.makedirs(label_used_dir)
    img_batch_dir = output_dir + '/batch_data'
    if not os.path.exists(img_batch_dir):
        os.makedirs(img_batch_dir)
    label_batch_dir = output_dir + '/batch_label'
    if not os.path.exists(label_batch_dir):
        os.makedirs(label_batch_dir)

    mean = np.zeros(len(image_subfixes), dtype=np.float64);
    mean_squar = np.zeros(len(image_subfixes), dtype=np.float64);
    n_samples = 0
    for fname in labellist:
        print fname
        day = fname.split(label_subfix)[0]
        inputs = []
        target = []
        hhv = []
        for subfix in image_subfixes:
            filein = sardir + "/" +  str(day) + subfix
            image = read_image(filein)
            hhv.append(image)
        hhv = np.asarray(hhv)
        labels = []
        labelpath = labeldir + "/" + fname
        label = read_image(labelpath)
        locs = gen_sample_locs(label, sample_rate, ignore_value, int(window/2) + 1);
        inputs, labels, subpoints = patch_img_label(hhv, label, locs, window)
        assert(len(inputs)==len(labels)==len(subpoints))
        #np.savetxt(label_used_dir + '/' + str(day)+'_label_used.txt', np.asarray(subpoints),fmt='%.2f')
        target = []
        # give dummy target for labels
        for i in range(len(inputs)):
            target.append(-1)
        outname = img_batch_dir + '/' + str(day)+'.batch'
        export_proto(inputs,target, outname)
        outname = label_batch_dir + '/' + str(day)+'.batch'
        labels = [np.expand_dims(l, axis = 0) for l in labels ]
        export_proto(labels, target, outname)

        n_samples += len(target);
        sub_total = [np.sum(data, axis = (1,2) ) for data in inputs]
        sub_total_squar = [np.sum(data.astype(np.int32)*data.astype(np.int32), axis = (1,2) ) for data in inputs]
        mean += np.sum(np.asarray(sub_total), axis = 0)
        mean_squar += np.sum(np.asarray(sub_total_squar), axis = 0 )
    mean = mean / n_samples / window / window
    std = np.sqrt(mean_squar / n_samples / window / window - mean * mean)
    stats_file = output_dir + '/mean_std.txt'
    f = open(stats_file,'w')
    np.savetxt(stats_file, np.concatenate((mean.reshape((2,1)), std.reshape(2,1)), axis = 1), fmt='%.2f')

def gen_ima_sample(window, sardir, image_subfixes, imadir, ima_subfix, output_dir):
    """
        generate ice concentration samples from image analysis
    """
    imalist = os.listdir(imadir)
    for fname in imalist:
        if not fname.endswith(ima_subfix):
            imalist.remove(fname)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ima_used_dir = output_dir + '/ima_used'
    if not os.path.exists(ima_used_dir):
        os.makedirs(ima_used_dir)
    batch_dir = output_dir + '/batch'
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)

    mean = np.zeros(len(image_subfixes), dtype=np.float64);
    mean_squar = np.zeros(len(image_subfixes), dtype=np.float64);
    n_samples = 0
    for fname in imalist:
        print fname
        day = fname.split(ima_subfix)[0]
        inputs = []
        target = []
        outname = batch_dir + '/' + str(day) + '.batch'
        hhv = []
        for subfix in image_subfixes:
            filein = sardir + "/" +  str(day) + subfix
            image = read_image(filein)
            hhv.append(image)
        hhv = np.asarray(hhv)
        ima = []
        with open(imadir+"/"+fname) as f:
            for line in f:
                point = map(float,line.split(' '))
                # mutiply ice concentration by 10
                point[2] = point[2] * 10
                ima.append(point)
        inputs, target, subpoints = patch_img(hhv, ima, window)
        #inputs, target, subpoints = patch_img_label_multi_scale(hhv, 0.25, ima, window)
        assert(len(inputs)==len(target)==len(subpoints))
        np.savetxt(ima_used_dir + '/' + str(day)+'_ima_used.txt', np.asarray(subpoints),fmt='%.2f')
        export_proto(inputs,target, outname)

        n_samples += len(target);
        sub_total = [np.sum(data, axis = (1,2) ) for data in inputs]
        sub_total_squar = [np.sum(data.astype(np.int32)*data.astype(np.int32), axis = (1,2) ) for data in inputs]
        mean += np.sum(np.asarray(sub_total), axis = 0)
        mean_squar += np.sum(np.asarray(sub_total_squar), axis = 0 )
    mean = mean / n_samples / window / window
    std = np.sqrt(mean_squar / n_samples / window / window - mean * mean)
    stats_file = output_dir + '/mean_std.txt'
    f = open(stats_file,'w')
    np.savetxt(stats_file, np.concatenate((mean.reshape((2,1)), std.reshape(2,1)), axis = 1), fmt='%.2f')

def gen_ima_patch_sample(window, sardir, image_subfixes, scale, imadir, ima_subfix, label_noise_std, coarse_predict_dir, coarse_predict_subfix, boundary, mask_dir, output_dir):
    imalist = os.listdir(imadir)
    for fname in imalist:
        if not fname.endswith(ima_subfix):
            imalist.remove(fname)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ima_used_dir = output_dir + '/ima_used'
    if not os.path.exists(ima_used_dir):
        os.makedirs(ima_used_dir)
    batch_dir = output_dir + '/batch'
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    tmp_label_dir = output_dir + '/tmp_label'
    if not os.path.exists(tmp_label_dir):
        os.makedirs(tmp_label_dir)
    label_dir = output_dir + '/label'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    mean = np.zeros(len(image_subfixes) + int(coarse_predict_dir != ""), dtype=np.float64);
    mean_squar = np.zeros(len(image_subfixes) + int(coarse_predict_dir != ""), dtype=np.float64);
    n_samples = 0
    for fname in imalist:
        print fname
        day = fname.split(ima_subfix)[0]
        inputs = []
        target = []
        outname = batch_dir + '/' + day +'.batch'
        hhv = []
        for subfix in image_subfixes:
            filein = sardir + "/" +  day + subfix
            image = read_image(filein)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)
            hhv.append(image)
        if coarse_predict_dir != "":
            image = read_image(coarse_predict_dir + "/" + day + coarse_predict_subfix )
            image = cv2.resize(image, (hhv[0].shape[1], hhv[0].shape[0]), interpolation = cv2.INTER_LINEAR )
            hhv.append(image)
        hhv = np.asarray(hhv)
        mask = cv2.imread(mask_dir + '/' + day + '-mask.tif', -1)
        mask = cv2.resize(mask, (hhv[0].shape[1], hhv[0].shape[0]), interpolation = cv2.INTER_NEAREST)
        ima = np.loadtxt(imadir + "/" + fname)
        ima[:,2] = ima[:,2] * 10
        # ima to labels initialize with 11
        ima[:,0:2] = ima[:,0:2] * scale
        label = np.zeros((hhv.shape[1], hhv.shape[2]), dtype=np.uint8)
        gx, gy = np.mgrid[0:hhv.shape[1], 0:hhv.shape[2]]
        locs = np.asarray([ima[:,1], ima[:,0]]).T
        label = griddata(locs, ima[:,2], (gx, gy), method='nearest')
        label = label.astype(np.uint8, copy=False)
        # add label noise for testing
        if label_noise_std > 0:
            noise = np.random.normal(0,label_noise_std,label.shape);
            index = (label>0) & (label < 10);
            noise = (noise + label) * index;
            noise[noise > 10] = 10;
            noise[noise < 0] = 0;
            label[index] = noise[index];
        label[mask != 0] = 11
        ima_valid = [p for p in ima if p[0] > boundary and p[0] < hhv.shape[2] - boundary and p[1] > boundary and p[1] < hhv.shape[1] - boundary and mask[p[1], p[0]] == 0 ]
        #for p in ima_valid:
        #    label[p[1], p[0]] = p[2]
        Image.fromarray(label).save(tmp_label_dir + "/" + str(day) + '.tif')
        inputs, target, subpoints = patch_img_label(hhv, label, ima_valid, window)
        assert(len(inputs)==len(target)==len(subpoints))
        np.savetxt(ima_used_dir + '/' + str(day) + '_' + str(scale) + '_ima_used.txt', np.asarray(subpoints),fmt='%.2f')
        point_target = [int(p[2]) for p in subpoints]
        outname = batch_dir + '/' + str(day) + '_' + str(scale) + '.batch'
        export_proto(inputs, point_target, outname)
        outname = label_dir + '/' + str(day) + '_' + str(scale) + '.batch'
        target = [np.expand_dims(l, axis = 0) for l in target ]
        export_proto(target, point_target, outname)

        n_samples += len(target);
        sub_total = [np.sum(data, axis = (1,2) ) for data in inputs]
        sub_total_squar = [np.sum(data.astype(np.int32)*data.astype(np.int32), axis = (1,2) ) for data in inputs]
        mean += np.sum(np.asarray(sub_total), axis = 0)
        mean_squar += np.sum(np.asarray(sub_total_squar), axis = 0 )
    mean = mean / n_samples / window / window
    std = np.sqrt(mean_squar / n_samples / window / window - mean * mean)
    stats_file = output_dir + '/mean_std_' + str(scale ) + '.txt'
    f = open(stats_file,'w')
    np.savetxt(stats_file, np.concatenate((mean.reshape((mean.size,1)), std.reshape(std.size,1)), axis = 1), fmt='%.2f')

def gen_ima_patch_sample_stage_2(window, hhv_dir, hhv_subfix, predict_local_dir, predict_global_dir, predict_subfix, imadir, ima_subfix, output_dir):
    imalist = os.listdir(imadir)
    for fname in imalist:
        if not fname.endswith(ima_subfix):
            imalist.remove(fname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    batch_dir = output_dir + '/batch_stage2'
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)

    mean = np.zeros(len(hhv_subfix), dtype=np.float64);
    mean_squar = np.zeros(len(hhv_subfix), dtype=np.float64);
    n_samples = 0
    for fname in imalist:
        print fname
        day = fname.split(ima_subfix)[0]
        inputs = []
        target = []
        outname = batch_dir + '/' + str(day)+'.batch'
        hhv = []
        #filein = predict_local_dir + "/" +  str(day) + predict_subfix
        #image = read_image(filein)
        for subfix in hhv_subfix:
            filein = hhv_dir + "/" + str(day) + subfix;
            image = read_image(filein)
            hhv.append(image)
        filein = predict_global_dir + "/" +  str(day) + predict_subfix
        image = read_image(filein)
        hhv.append(image)
        hhv = np.asarray(hhv)
        ima = np.loadtxt(imadir + "/" + fname)
        ima[:,2] = ima[:,2] * 10
        label = np.zeros((hhv.shape[1], hhv.shape[2]), dtype=np.uint8) + 100
        ima_valid = [p for p in ima if p[0] > 0 and p[0] < hhv.shape[2] and p[1] > 0 and p[1] < hhv.shape[1]]
        for p in ima_valid:
            label[p[1], p[0]] = p[2]
        inputs, target, subpoints = patch_img_label(hhv, label, ima, window)
        assert(len(inputs)==len(target)==len(subpoints))
        point_target = [int(p[2]) for p in subpoints]
        outname = batch_dir + '/' + str(day) + '.batch'
        export_proto(inputs, point_target, outname)

        n_samples += len(target);
        sub_total = [np.sum(data, axis = (1,2) ) for data in inputs]
        sub_total_squar = [np.sum(data.astype(np.int32)*data.astype(np.int32), axis = (1,2) ) for data in inputs]
        mean += np.sum(np.asarray(sub_total), axis = 0)
        mean_squar += np.sum(np.asarray(sub_total_squar), axis = 0 )
    mean = mean / n_samples / window / window
    std = np.sqrt(mean_squar / n_samples / window / window - mean * mean)
    stats_file = output_dir + '/mean_std.txt'
    f = open(stats_file,'w')
    np.savetxt(stats_file, np.concatenate((mean.reshape((len(hhv_subfix),1)), std.reshape(len(hhv_subfix),1)), axis = 1), fmt='%.2f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir")
    parser.add_argument("--image_subfixes")
    parser.add_argument("--scale")
    parser.add_argument("--ima_dir")
    parser.add_argument("--ima_subfix")
    parser.add_argument("--label_dir")
    parser.add_argument("--label_subfix")
    parser.add_argument("--sample_density")
    parser.add_argument("--patch_size", default=45)
    parser.add_argument("--coarse_predict_dir", default = '') # used in second stage
    parser.add_argument("--coarse_predict_subfix", default = '')
    parser.add_argument("--margin", default = 0)
    parser.add_argument("--output_dir")
    parser.add_argument("--mask_dir", default = 0)
    parser.add_argument("--sample_rate", default = 0.1)
    parser.add_argument("--ignore_value", default = -1)
    parser.add_argument("--label_noise_std", default = -1)
    args = parser.parse_args()
    sardir = args.image_dir
    image_subfixes = args.image_subfixes.strip('\"').split(',')
    scale = float(args.scale)
    imadir = args.ima_dir
    ima_subfix = args.ima_subfix
    #labeldir = args.label_dir
    #label_subfix = args.label_subfix
    #nsample_per_label = int(args.sample_density)
    window = int(args.patch_size)
    output_dir = args.output_dir
    gen_ima_patch_sample(window, sardir, image_subfixes, scale, imadir, ima_subfix, args.label_noise_std, args.coarse_predict_dir, args.coarse_predict_subfix, int(args.margin),args. mask_dir, output_dir)

    #gen_ima_sample(window, sardir, image_subfixes, imadir, ima_subfix, output_dir)
    #gen_fcn_sample(window, sardir, image_subfixes, args.label_dir, args.label_subfix, float(args.sample_rate), int(args.ignore_value), args.output_dir)

    #window = 45
    #predict_local_dir = '/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_stage_1_45/predict'
    #predict_global_dir = '/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_stage_1_45/predict25_rescale'
    #predict_subfix = '.tif'
    #gen_ima_patch_sample_stage_2(window, sardir, image_subfixes, predict_local_dir, predict_global_dir, predict_subfix, imadir, ima_subfix, output_dir)

#! /usr/bin/python
import numpy as np
import caffe
import sys
import argparse
import cv2
import re
import string
import os

PREDICT_SCALE = 100
IMA_SCALE = 10

def date_from_path(fname):
    fname = fname.split('/')[-1]
    p_gsl = re.compile("\d{8,8}_\d{6,6}")
    if p_gsl.match(fname) != None:
        return fname[0:15]
    else:
        return fname[0:8]

def export_latex(fname, stats_train, stats_valid, stats_test):
    # make sure fname end with .tex
    # X is a matrix with each row is the error evaluation for one configuration
    tex_file = open(fname,'w')
    s = '''\\begin{table}[h]
        \\centering
        \\caption{The error statistics for train or test or validataion datasets.}
        \\begin{tabular}{ccccccc}
        \\toprule
        Set & Scene & Num & $E_{sig}$ & $E_{L1}$ & Std & MSE \\\\\n'''
    tex_file.write(s)
    for strset, stats in zip(['train', 'valid', 'test'], [stats_train, stats_valid, stats_test]):
        s = '''\\midrule\n'''
        tex_file.write(s)
        s = '\\multirow{%d}{*}{%s}' % (len(stats), strset)
        tex_file.write(s)
        for e in stats:
            day = e['day']
            day = string.replace(day, '_', '\\_')
            s = '\t&' + day + '\t&' + str(e['num']) + ' \t& %.4f\t& %.4f\t& %.4f\t& %.4f\\\\\n' % (e['avg'], e['abs_avg'], e['std'], e['mse'] )
            tex_file.write(s)
    s = '''\\bottomrule
        \\end{tabular}
        \\label{table:table_errors}
        \\end{table}\n'''
    tex_file.write(s)
    tex_file.close()

def load_batch(batch_path, crop_size):
    batch = caffe.proto.caffe_pb2.DatumVector()
    batch.ParseFromString(open(batch_path).read())
    if crop_size == 0:
        crop_size = batch.datums[0].height
    offset = (batch.datums[0].height - crop_size) / 2
    datums = [caffe.io.datum_to_array(d)[:, offset:offset+crop_size, offset:offset+crop_size] for d in batch.datums ]
    labels = [float(d.label) for d in batch.datums]
    return (datums, labels)

def cal_statistics(diff):
    print 'num: %.4f\t avg: %.4f\t abs_avg: %.4f\t std:%.4f\t mse:%.4f' % (len(diff), diff.mean(), np.absolute(diff).mean(), diff.std(), np.mean(diff**2) )

def main_eval(arch_path, weights_path, mean_std_path, batch_path, out_path):
    try:
        caffe.set_mode_gpu()
    except:
        pass

    classifier = caffe.Classifier(arch_path, weights_path)
    try:
        classifier.set_mode_gpu()
    except:
        pass

    mean_std = np.loadtxt(mean_std_path)
    avg = mean_std[:,0].reshape(mean_std.shape[0], 1, 1)
    dev = mean_std[:,1].reshape(mean_std.shape[0], 1, 1)
    predict = []
    label = []
    stats = []
    for path in batch_path:
        print path
        day = date_from_path(path)
        batch = load_batch(path, classifier.image_dims[0])
        l = [a / 10 for a in batch[1]]
        label.extend(l)
        batch_data = [(b - avg) / dev for b in batch[0]]
        batch_data = [b.transpose( (1,2,0) ) for b in batch_data]
        p = []
        batch_size = 128
        idx = 0
        while idx < len(batch_data):
            p.extend(classifier.predict(batch_data[idx : min(len(batch_data), idx + batch_size)], oversample=False).flatten().tolist())
            idx += batch_size
        print "%d,%d" (len(p), len(batch_data))
        assert(len(p) == len(batch_data))
        np.savetxt(path[0 : path.rfind('.')] + '.predict.txt', p, fmt='%.4f')
        predict.extend(p)
        assert(len(p) == len(l))
        diff = np.asarray(p).flatten() - np.asarray(l).flatten()
        cal_statistics(diff)
        stats.append( {'num': len(diff), 'day': day, 'avg': diff.mean(), 'std': diff.std(), 'abs_avg': np.absolute(diff).mean(), 'mse': np.mean(diff**2)} )
    np.savetxt(out_path, np.concatenate((np.expand_dims(predict, axis = 1), np.expand_dims(label, axis = 1) ), axis = 1), fmt = '%.4f')
    diff = np.asarray(predict).flatten() - np.asarray(label).flatten()
    cal_statistics(diff)
    stats.append( {'num': len(diff), 'day': 'average', 'avg': diff.mean(), 'std': diff.std(), 'abs_avg': np.absolute(diff).mean(), 'mse': np.mean(diff**2)} )
    return stats

def eval_from_image(predict_path, scale, margin, mask_path, ima_path, out_path):
    assert(len(predict_path) == len(mask_path) )
    assert(len(predict_path) == len(ima_path) )
    data = []
    stats = []
    for im_p, mask_p, ima_p in zip(predict_path, mask_path, ima_path):
        assert(os.path.isfile(im_p)), im_p
        assert(os.path.isfile(mask_p)), mask_p
        assert(os.path.isfile(ima_p)), ima_p
        im = cv2.imread(im_p, -1)
        #im = cv2.resize(im, None, fx = 1/scale, fy = 1/scale, interpolation = cv2.INTER_NEAREST)
        mask = cv2.imread(mask_p, -1)
        ima = np.loadtxt(ima_p)
        H = im.shape[0]
        W = im.shape[1]
        day = date_from_path(im_p)
        ima = [p for p in ima if p[0] > margin and p[0] < W - margin and p[1] > margin and p[1] < H - margin ]
        d = [[p[0], p[1], float(im[p[1], p[0]]) / PREDICT_SCALE, p[2] / IMA_SCALE ] for p in ima if mask[p[1], p[0]] == 0]
        diff = np.asarray([a[2] - a[3] for a in d]).flatten()
        cal_statistics(diff)
        stats.append( {'num': len(d), 'day': day, 'avg': diff.mean(), 'std': diff.std(), 'abs_avg': np.absolute(diff).mean(), 'mse': np.mean(diff**2)} )
        data.extend(d)
        np.savetxt(out_path[0 : out_path.rfind('/') + 1] + day + '.predict.txt', d, fmt='%.4f')
    diff = np.asarray([a[2] - a[3] for a in data]).flatten()
    cal_statistics(diff)
    np.savetxt(out_path, data, fmt = '%.4f')
    stats.append( {'num': len(diff), 'day': 'average', 'avg': diff.mean(), 'std': diff.std(), 'abs_avg': np.absolute(diff).mean(), 'mse': np.mean(diff**2) } )
    return stats

def main_eval_image(predict_dir, scale, margin, predict_file_list, mask_dir, ima_dir, out_path):
    predict_path = open(predict_file_list).readlines()
    mask_path = [mask_dir + '/' + p[0:p.rfind('_')] + '-mask.tif' for p in predict_path]
    ima_path = [ima_dir + '/' + p[0:p.rfind('_')] + '_' +  str(scale) + '_ima_used.txt' for p in predict_path]
    predict_path = [predict_dir + '/' + p[0:p.rfind('_')] + '.tif' for p in predict_path ]
    return eval_from_image(predict_path, scale, margin, mask_path, ima_path, out_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch")
    parser.add_argument("--weights")
    parser.add_argument("--mean_std")
    parser.add_argument("--train_source")
    parser.add_argument("--test_source")
    parser.add_argument("--valid_source")
    parser.add_argument("--out")
    parser.add_argument("--predict_dir")
    parser.add_argument("--mask_dir")
    parser.add_argument("--ima_dir")
    parser.add_argument("--margin")
    parser.add_argument("--scale")

    args = parser.parse_args()
    stats = {}
    if args.arch:
        folder = args.train_source[0:args.train_source.rfind('/')]
        print folder
        for strset, batch_file in zip(['valid', 'test', 'train'], [args.valid_source, args.test_source, args.train_source] ):
            print strset
            batch_path = open(batch_file).readlines()
            batch_path = [folder + '/' + l.strip() for l in batch_path]
            out_path = args.out[0:args.out.rfind('.')] + '_' + strset + '.txt';
            stats[strset] = main_eval(args.arch, args.weights, args.mean_std, batch_path, out_path)
    elif args.predict_dir:
        for strset, batch_file in zip(['train', 'valid', 'test'], [args.train_source, args.valid_source, args.test_source] ):
            print strset
            out_path = args.out[0:args.out.rfind('.')] + '_' + strset + '.txt';
            stats[strset] = main_eval_image(args.predict_dir, float(args.scale), float(args.margin), batch_file, args.mask_dir, args.ima_dir, out_path)
    tex_path = args.out[0:args.out.rfind('.')] + '_table.tex'
    export_latex(tex_path, stats['train'], stats['valid'], stats['test'])

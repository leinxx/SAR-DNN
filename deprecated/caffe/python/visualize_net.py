

import numpy as np
import caffe


def load_data(fiein, mean_std_file):
    lines = open(mean_std_file).readlines()
    lines = [l.strip().spit() for l in lines]
    l = np.asarray(lines).astype(np.float32)
    mean = l[:,0]
    std = l[:,1]
    f = open(filein)
    batch = caffe.proto.caffe_pb2.DatumVector()
    batch.ParseFromString(f.read())
    arr = [((np.transpose(datum_to_array(d).astype(np.float32), (1, 2, 0)) - mean) / std, d.label) for d in batch.datums]
    return arr


def pattern(net, data, K):
    '''
    the largest K samples for each unit of the net
    '''
    pass

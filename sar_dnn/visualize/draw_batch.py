#! /usr/bin/python
import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys

def draw(batchfile, start, end, ext, outdir):
    batch = caffe.proto.caffe_pb2.DatumVector()
    batch.ParseFromString(open(batchfile).read())
    batch = [caffe.io.datum_to_array(b) for b in batch.datums]
    for i, b in zip(range(start,end), batch[start:end]):
        for c in range(b.shape[0]):
            plt.imshow(b[c,:,:], cmap='gray')
            plt.savefig(outdir+ '/' + str(i) + '_' + str(c) + '_' + ext + '.png')

if __name__ ==  '__main__':
    draw(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])


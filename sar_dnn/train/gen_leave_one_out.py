#! /usr/bin/python

import numpy as np
import os

if __name__ == '__main__':
    targetdir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_65/ima_used/'
    flist = os.listdir(targetdir)
    n_samples = []
    for l in flist:
        with open(targetdir + '/' + l) as f:
            data = f.readlines()
            n_samples.append(len(data))
    flist = [l[0:15]+'_1.0.batch' for l in flist if l.endswith('.txt')]
    flist = np.asarray(flist)
    n_samples = np.asarray(n_samples)

    print "%d of batch files found" % len(flist)

    for r in range(0,10):
        #determine testing and validation, the rest for training
        idx = np.arange(len(flist))
        np.random.shuffle(idx)
        test = flist[idx[0:4]]
        valid = flist[idx[4:8]]
        train = flist[idx[8:]]
        with open(targetdir + '../batch/train_source_lou_' + str(r) + '.txt', 'w') as f:
            f.write('\n'.join(train))
        with open(targetdir + '../batch/test_source_lou_' + str(r) + '.txt', 'w') as f:
            f.write('\n'.join(test))
        with open(targetdir + '../batch/valid_source_lou_' + str(r) + '.txt', 'w') as f:
            f.write('\n'.join(valid))



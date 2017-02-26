#!/usr/bin/env python
# plot loss from caffe training log file
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import sys
if __name__ == "__main__":
    if len(sys.argv) == 1:
        logfile = 'log.txt'
    else:
        logfile = sys.argv[1];
    lines = open(logfile).readlines()
#extract test
    test_it_lines = [l.strip().split() for l in lines if l.find('Testing net (#0)') != -1]
    test_it = [l[-4].split(',')[0] for l in test_it_lines]
    test_loss_lines = [l.strip().split() for l in lines if l.find('Test net output #0') != -1]
    test_loss = [l[-2] for l in test_loss_lines]
    test_it = np.asarray(test_it).astype(int)
    test_loss = np.asarray(test_loss).astype(float)
    test_it = test_it[0:len(test_loss)]
#test_loss_lines_1 = [l.strip().split() for l in lines if l.find('Test net output #1') != -1]
#test_loss_1 = [l[-2] for l in test_loss_lines_1]
#test_loss_1 = np.asarray(test_loss_1).astype(float)


#extract test
    lines = [l for l in lines if l.find('Test net output #') == -1]
    train_lines = [l.strip().split() for l in lines if l.find(' loss = ') != -1]
    train_loss = [l[-1] for l in train_lines[1:]]
    train_it = [l[-4].strip(',') for l in train_lines[1:]]
    train_it = train_it[0:len(train_loss)]
    train_loss = np.asarray(train_loss).astype(float);
    train_it = np.asarray(train_it).astype(int);
#train_loss = np.sqrt(train_loss/40);
#test_loss = np.sqrt(test_loss/10);
    for i,l in zip(train_it, train_loss):
        print 'train_iter %d \t loss %f' % (i, l)
    for i,l in zip(test_it, test_loss):
        print 'test_iter %d \t loss %f' % (i, l)
    plt.plot(train_it, train_loss, label='train')
    plt.plot(test_it, test_loss, label='test_0')
    np.savetxt("train_loss.txt", np.asarray([train_it, train_loss]).T)
    np.savetxt("test_loss.txt", np.asarray([test_it, test_loss]).T )
#plt.plot(test_it, test_loss_1, label='test_1')
#plt.plot(test_it, test_loss_1 + test_loss, label='test')
    plt.legend()
    plt.show()


import numpy as np
import cv2
import caffe
from evaluate import load_batch


def activation(batch, classifier):
    '''
    get all the activation of the input list
    return: [[layer: activation] ]
    '''
    pass

def activation_from_file(batch_path, arch_path, weight_path, mean_std_path):
    caffe.set_mode_gpu()
    classifier = caffe.Classifier(arch_path, weight_path)
    blobs = classifier.blobs.keys()
    blobs = [k for k in blobs if k.find('pool') == -1] # no pooling layer
    mean_std = np.loadtxt(mean_std_path)
    avg = mean_std[:,0].reshape(mean_std.shape[0], 1, 1)
    dev = mean_std[:,1].reshape(mean_std.shape[0], 1, 1)
    data = []
    for path in batch_path:
        batch = load_batch(path, classifier.image_dims[0])
        batch_data = [(b - avg) / dev for b in batch[0]]
        data.extend([b.transpose( (1,2,0) ) for b in batch_data])
    out = classifier.extract_features(data, blobs = blobs)
    return out


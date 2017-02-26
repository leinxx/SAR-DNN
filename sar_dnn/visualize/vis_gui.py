#!/usr/bin/env python
import os
import sys
import numpy as np
import argparse
from PyQt4 import QtCore, QtGui
from vis import activation_from_file
from evaluate import load_batch
import cv2
def array_to_pixmap(im):
    QI=QtGui.QImage(im.data, im.shape[1], im.shape[0], im.shape[1], QtGui.QImage.Format_Indexed8)
    return QtGui.QPixmap.fromImage(QI)

class ImageFileList(QtGui.QListWidget):
    def __init__(self, images, labels, predicts):
        QtGui.QListWidget.__init__(self)
        self._images = images
        self._labels = labels
        self._predicts = predicts
        if len(predicts) == 0:
            self._predicts = labels
        self._populate()
    def _populate(self):
        for image, label, predict in zip(self._images, self._labels, self._predicts):
            item = QtGui.QListWidgetItem(self)
            item.setText('%d, %.2f' % (label, predict * 10) )
            image = cv2.resize(image[0,:,:], (100,100), interpolation = cv2.INTER_NEAREST)
            item.setIcon(QtGui.QIcon(array_to_pixmap(image)))
            self.addItem(item)

class ImageList(QtGui.QWidget):
    def __init__(self, images):
        super(ImageList, self).__init__()
        self._images = images[0:10,:,:,:] # N * C * H * W
        self.initUI()
    def initUI(self):
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self._image_widgets = []
        for i, image in zip(range(self._images.shape[0]), self._images):
            self._image_widgets.append(QtGui.QLabel())
            self._image_widgets[i].setPixmap(array_to_pixmap(image[0,:,:]))
            self._image_widgets[i].setAccessibleName(str(i))
            layout.addWidget(self._image_widgets[i])

    def set_image(images):
        self._images = images

    def set_view_widget(widget):
        self._view_widget = widget

def feature_to_image(feature):
    if feature.ndim == 2:
        return feature

    H = int(np.ceil(np.sqrt(feature.shape[0])))
    sub_h = feature.shape[1]
    gap = 5
    im = np.zeros((H * (sub_h + gap), H * (sub_h + gap))) + 20
    for i in range(feature.shape[0]):
        row = int(i / H)
        col = i % H
        r = row * (sub_h + gap)
        c = col * (sub_h + gap)
        im[r: r + sub_h, c:c + sub_h ] = feature[i,:,:]# / 2 + 122
        print 'min: {0}, max: {1}'.format(feature[i,:,:].min(), feature[i,:,:].max());

    return im

class vis(QtGui.QWidget):
    def __init__(self, batch_path, arch_path, weight_path, mean_std_path):
        super(vis, self).__init__()
        self._n = int(0)
        self._c = int(0)
        assert(os.path.isfile(batch_path) )
        self._batch_path = batch_path
        self._arch_path = arch_path
        self._weight_path = weight_path
        self._mean_std_path = mean_std_path
        self._x, self._y = load_batch(batch_path, 0)
        self._x = np.asarray(self._x)
        feature_save_name = batch_path.split('/')[-1] + 'feature.npz'
        show_feature = arch_path != '' and weight_path != '' and mean_std_path != ''
        if os.path.isfile(feature_save_name):
            self._feature = np.load(feature_save_name)
            self._predict = self._feature['fc5']
        elif show_feature:
            self._feature = activation_from_file([self._batch_path], self._arch_path, self._weight_path, self._mean_std_path)
            np.savez_compressed(feature_save_name, **self._feature)
            self._predict = self._feature['fc5']
        else:
            self._feature = {'data': self._x}
            self._predict = []
        data = {}
        for key, value in self._feature.iteritems():
            value = 255 * (value.astype(float) - value.min()) / (value.max() - value.min())
            data[key] = value.astype(np.uint8)
        self._feature = data
        self.initUI()

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        self.lst = ImageFileList(self._x, self._y, self._predict)
        self.lst.setIconSize(QtCore.QSize(100,100))
        self.lst.setFixedSize(180,1000)
        #for i in self.lst._image_widgets:
        #    i.installEventFilter(self)
            #i.mousePressEvent = self.update_sample
        layout.addWidget(self.lst)
        self.lst.currentItemChanged.connect(self.update_sample)

        self.layerlst = QtGui.QListWidget()
        self.layerlst.setFixedSize(100, 1000)
        layout.addWidget(self.layerlst)
        for key in self._feature.keys():
            item = QtGui.QListWidgetItem()
            item.setText(key)
            self.layerlst.addItem(item)
        self.layerlst.currentItemChanged.connect(self.update_sample)
        self.image = QtGui.QLabel()
        #self.image.setScaledContents(True)
        self.image.setFixedSize(1000,1000)
        layout.addWidget(self.image)

        #self.setGeometry(100,100,600,800)
        self.setWindowTitle('vis')
        self.show()

    #def eventFilter(self, source, event):
    #    if (event.type() == QtCore.QEvent.MouseButtonPress and
    #        isinstance(source, QtGui.QLabel)):
    #        import pdb
    #        pdb.set_trace()
    #        n = int(source.accessibleName())
    #        print n

    def update_sample(self):# H * W
        n = self.lst.currentRow()
        if self.layerlst.currentItem() == None or n == None:
            return
        layer = str(self.layerlst.currentItem().text())
        print n, layer
        if self._feature[layer].ndim == 2:
            im = self._feature[layer]
        else:
            im = feature_to_image(self._feature[layer][n])
        im = cv2.resize(im, (1000,1000), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite('test.png', im)
        self.image.setPixmap(QtGui.QPixmap('test.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch')
    parser.add_argument('--arch', default = '')
    parser.add_argument('--weights',default = '')
    parser.add_argument('--meanfile',default = '')
    args = parser.parse_args()

    app = QtGui.QApplication([])
    ex = vis(args.batch, args.arch, args.weights, args.meanfile)
    sys.exit(app.exec_())


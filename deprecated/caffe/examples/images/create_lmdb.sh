TA=examples/images
rm -rf $DATA/img_train_lmdb
build/tools/convert_imageset.bin --shuffle \
  --resize_height=256 --resize_width=256 \
  /home/ein/lei-phd-20160912/dev/caffe/examples/images/ $DATA/train.txt  $DATA/img_train_lmdb

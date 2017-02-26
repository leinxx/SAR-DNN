arch=.fill_deploy_vgg.prototxt
basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_45/
batchdir=${basedir}/batch
snapshot=${basedir}/model_l2/_iter_8000.caffemodel

meanfile=${basedir}/mean_std.txt
trainsource=${batchdir}/train_source.txt
traintargetsource=${targetdir}/train_source.txt
trainmean=${meanfile}
testsource=${batchdir}/test_source.txt
testtargetsource=${targetdir}/test_source.txt
testmean=${meanfile}

gdb --args /home/lein/dev/caffe/build/tools/extract_features \
    $snapshot $arch conv1 test 128 lmdb GPU

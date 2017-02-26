# the solver used here is a concatanation of the caffe solver and caffe arch file
# a separator is added to indicat the seperation of their contents
# this script first seperate the combined file into two files: solver file and arch
# then the solver and arch files will be configured/filled 

solver_arch=solver_3layer_fc.prototxt
#solver_arch=solver_3layer_fc_stage_1.prototxt
cropsize=41
labelcropsize=1
basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_45
batchdir=${basedir}/batch
targetdir=${basedir}/label
modeldir=${basedir}/model
mkdir -p $modeldir
snapshot=${modeldir}/_iter_37000.solverstate

meanfile=${basedir}/mean_std.txt
trainsource=${batchdir}/train_source.txt
traintargetsource=${targetdir}/train_source.txt
trainmean=${meanfile}
testsource=${batchdir}/test_source.txt
testtargetsource=${targetdir}/test_source.txt
testmean=${meanfile}

meanfileg=${basedir}/mean_std.txt
trainsourceg=${batchdir}/train_source_g.txt
traintargetsourceg=${targetdir}/train_source_g.txt
trainmeang=${meanfileg}
testsourceg=${batchdir}/test_source_g.txt
testtargetsourceg=${targetdir}/test_source_g.txt
testmeang=${meanfileg}

# 1. split solver_arch file into solver file and arch file
solver=.solver_split.prototxt
arch=.arch_train_val_split.prototxt
lines=$(grep -n "##splitmark##" ${solver_arch} | grep -Eo '^[^:]+')
lines=$((lines - 1))
linestail=$(wc -l < ${solver_arch})
linestail=$((linestail - lines - 1))
head -n $lines ${solver_arch} > ${solver}
tail -n $linestail  ${solver_arch} > ${arch}

# 2. fill solver and arch with the correct setting
fancyDelim=$(printf '\001')

sed -i "s${fancyDelim}\$train_mean_g${fancyDelim}${trainmeang}${fancyDelim}g" $arch
sed -i "s${fancyDelim}\$train_source_g${fancyDelim}${trainsourceg}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$train_target_source_g${fancyDelim}${traintargetsourceg}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_mean_g${fancyDelim}${testmeang}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_source_g${fancyDelim}${testsourceg}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_target_source_g${fancyDelim}${testtargetsourceg}${fancyDelim}g" ${arch}

sed -i "s${fancyDelim}\$train_mean${fancyDelim}${trainmean}${fancyDelim}g" $arch
sed -i "s${fancyDelim}\$train_source${fancyDelim}${trainsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$train_target_source${fancyDelim}${traintargetsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_mean${fancyDelim}${testmean}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_source${fancyDelim}${testsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_target_source${fancyDelim}${testtargetsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$label_crop_size${fancyDelim}${labelcropsize}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$net${fancyDelim}${arch}${fancyDelim}g" ${solver}
sed -i "s${fancyDelim}\$snapshot${fancyDelim}${modeldir}/${fancyDelim}g" ${solver}


# 3. run
gdb --args /home/lein/dev/caffe/build/tools/caffe train \
    --solver=${solver} 2>&1 | tee log.txt
#gdb --args /home/lein/dev/caffe/build/tools/caffe train \
#    --solver=${solver} \
#    --snapshot=${snapshot} 2>&1 | tee log.txt
# 4. test
#gdb --args /home/lein/dev/caffe/build/tools/caffe test \
#    --model=${arch} --weights=${modeldir}/_iter_17000.caffemodel 2>&1 | tee log.txt


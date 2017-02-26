# Lei Wang, alphaleiw@gmail.com, Last modified Feb 26, 2017
# the solver used here is a concatanation of the caffe solver and caffe arch file.
# a separator is added to indicat the seperation of their contents
# this script does a few things in sequence to carry out a experiment:
# 1. define a experiment: set up solver path and experiment data path, snapshot path
# 2. Seperate the solver file to a solver file $solver a architecture file $arch
# 3. Configure the solver and arch files according to the experiment setting.
# 4. Train/test using the solver and architecture file

##### set up directories
exp_dir=../../../exp
basedir=$exp_dir/dataset/gsl2014_hhv_ima/batches_land_free_65
solverdir=../solver/train
deploydir=../solver/predict

# set experiment parameters
task=1 #1: train, 2: continue_train, 3: finetune, 4: test
channels=3
cropsize=45 #effective input patch size
labelcropsize=45 #45 # effective label size
resizelabel=12 # resize croped label to resizelabel for fcn
# loss type: L1 or L2
loss=L2 #L2, L1
scale=1.0 # keep it 1.0
finetune=0 # the number of layers to be frozen for finetune. 0 means finetune from layer 1 and up
label_noise=0 # add noise to training sample labels, useful for testing the sensitivity of the model to training sample noise
snapshot_iter=5000 # use this .solverstate file/model to initialize network paramters,  used when task=2
finetune_iter=5000 # use this .caffemodel file/model to initialize network parameters,  used when task=3

method=BASE # model: BASE or EM # or SUM_SPLIT
experiment=NONE #SQUEEZE #EVA_RANDOM_NOISE # experiment to run
if [ "$experiment" = EVA_RANDOM_NOISE ] # evaluate the effect of random noise in labels
then
  solver_arch=solver_base_fcn.prototxt
  deploy_arch=deploy_base_fcn.prototxt
  snapshotdir=model_${method}_${scale}_${loss}
  basedir=$exp_dir/dataset/beaufort_2010_2011/batches_${cropsize}_label_noise_$1
  label_noise=$1
elif [ "$experiment" = FINETUNE_EXP ] # fine tuning experiment
then
  solver_arch=solver_base_fcn_finetune.prototxt
  deploy_arch=deploy_base_fcn.prototxt
  finetune=$1
  snapshotdir=model_${method}_${scale}_${loss}_finetune_${finetune}
  method=BASE # method base is used for the evaluation of finetuning in thesis
elif [ "$experiment" = FUSE_ASI ] # use asi with HH and HV and IA to train the network
then
  solver_arch=solver_base_fcn_asi.prototxt
  deploy_arch=deploy_base_fcn_asi.prototxt
  snapshotdir=model_${method}_${scale}_${loss}_fuse_asi
  method=BASE # method base is used for fuse asi data
elif [ "$experiment" = SQUEEZE ] # use squeeze net
then
  solver_arch=squeeze_net.prototxt
  deploy_arch=squeeze_net_deploy.prototxt
  labelcropsize=1
  resizelabel=1
  snapshotdir=model_${method}_${scale}_${loss}_${experiment}
else # standard experiment
  solver_arch=solver_base_fcn.prototxt
  deploy_arch=deploy_base_fcn.prototxt
  #snapshotdir=model_${method}_${scale}_${loss}_lou_${1}
  snapshotdir=model_${method}_${scale}_${loss}
fi
solver_arch=$solverdir/$solver_arch
deploy_arch=$deloydir/$deploy_arch
snapshotdir=${basedir}/${snapshotdir}

##### make experiment directory to save trained models
mkdir -p $snapshotdir

# continue_training params, set task=2 to use
snapshot=${snapshotdir}/_iter_${snapshot_iter}.solverstate

# finetune / test params, set task=3 to use
modeldir=$exp_dir/dataset/gsl2014_hhv_ima/batches_land_free_65/model_base_l1/
weights=${modeldir}/_iter_${finetune_iter}.caffemodel


#### set image batches and label batches for caffe
### TODO (LEI): combine image batches and label batches to one file
batchdir=${basedir}/batch
targetdir=${basedir}/label

#### set up the mean value file and the the split of training/testing of samples, only batch files listed in the trainsource will be used for training. Check ../caffe/src/caffe/layers/datum_data_layer.cpp to see how trainsource is used. 1.0 is the scale of the training samples.
meanfile=${basedir}/mean_std_${scale}.txt
trainsourcedir=${batchdir}
trainsource=${batchdir}/train_source_1.0.txt
trainsourcecp=${snapshotdir}/train_source_1.0.txt
cp $trainsource ${trainsourcecp} # copy the configuration file to new location and use it, save everything about the experiment
trainsource=${trainsourcecp}
traintargetsource=${targetdir} #/train_source_${scale}.txt
trainmean=${meanfile}

testsourcedir=${batchdir}
testsource=${batchdir}/test_source_1.0.txt
testsourcecp=${snapshotdir}/test_source_1.0.txt
cp $testsource $testsourcecp
testsource=$testsourcecp
testtargetsource=${targetdir} #/test_source_${scale}.txt
testmean=${meanfile}


meanfileg=${basedir}/mean_std_1.0.txt
trainsourceg=${batchdir}/train_source_g.txt
traintargetsourceg=${targetdir}/train_source_g.txt
trainmeang=${meanfileg}
testsourceg=${batchdir}/test_source_g.txt
testtargetsourceg=${targetdir}/test_source_g.txt
testmeang=${meanfileg}

# 1. split solver_arch file into solver file and arch file
solver=${snapshotdir}/solver_split.prototxt
arch=${snapshotdir}/arch_train_val_split.prototxt
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

sed -i "s${fancyDelim}\$train_source_dir${fancyDelim}${trainsourcedir}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$train_mean${fancyDelim}${trainmean}${fancyDelim}g" $arch
sed -i "s${fancyDelim}\$train_source${fancyDelim}${trainsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$train_target_source${fancyDelim}${traintargetsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_source_dir${fancyDelim}${testsourcedir}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_mean${fancyDelim}${testmean}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_source${fancyDelim}${testsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$test_target_source${fancyDelim}${testtargetsource}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$label_crop_size${fancyDelim}${labelcropsize}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$resize_label${fancyDelim}${resizelabel}${fancyDelim}g" ${arch}
sed -i "s${fancyDelim}\$net${fancyDelim}${arch}${fancyDelim}g" ${solver}
sed -i "s${fancyDelim}\$snapshot${fancyDelim}${snapshotdir}/${fancyDelim}g" ${solver}
sed -i "s${fancyDelim}\$loss${fancyDelim}${loss}${fancyDelim}g" ${solver}
sed -i "s${fancyDelim}\$method${fancyDelim}${method}${fancyDelim}g" ${solver}



### set the bottom $finetune number of layers to have learning rate 0
for idx in {1..5}
do
  if [ "$idx" -lt "$finetune" ]
  then
    sed -i "s${fancyDelim}\$lr_${idx}_1${fancyDelim}0${fancyDelim}g" ${arch}
    sed -i "s${fancyDelim}\$lr_${idx}_2${fancyDelim}0${fancyDelim}g" ${arch}
  else
    sed -i "s${fancyDelim}\$lr_${idx}_1${fancyDelim}1${fancyDelim}g" ${arch}
    sed -i "s${fancyDelim}\$lr_${idx}_2${fancyDelim}2${fancyDelim}g" ${arch}
  fi
done
# back up files to the working folder and execute there
cp ${0} $snapshotdir/

# generate file for prediction
deployfill=$snapshotdir/deploy.prototxt
cp ${deploy_arch} ${deployfill}
fancyDelim=$(printf '\001')
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" $deployfill
sed -i "s${fancyDelim}\$channels${fancyDelim}${channels}${fancyDelim}g" $deployfill
predict_sh=${snapshotdir}/predict.sh
cp predict_template.sh $predict_sh
sed -i "s${fancyDelim}\$arc${fancyDelim}${deployfill}${fancyDelim}g" $predict_sh
sed -i "s${fancyDelim}\$channels${fancyDelim}${channels}${fancyDelim}g" $predict_sh

caffe=../caffe/build/tools/caffe
caffe_predict=${caffe}_predict
sed -i "s${fancyDelim}\$caffe_predict${fancyDelim}${caffe_predict}${fancyDelim}g" $predict_sh

cd $snapshotdir
#continue training
if [ $task -eq 1 ]
then
  # 3. train
  LOG=${snapshotdir}/log.txt
  echo "train"
  $caffe train \
        --solver=${solver} 2>&1 | tee $LOG
elif [ $task -eq 2 ] 
  # continue training
then
  echo "continue training"
  LOG=${snapshotdir}/log.txt
  $caffe train \
        --solver=${solver} \
        --snapshot=${snapshot} 2>&1 | tee -a $LOG
elif [ $task -eq 3 ]
then
  #fine tune
  echo "fine tune"
  #LOG=log_finetune_${modeldir##*/}_${weights##*/}.txt
  LOG=$snapshotdir/log.txt
  $caffe train \
        --solver=${solver} --weights=${weights} 2>&1 | tee $LOG
elif [ $task -eq 4 ]
then
  # 4. test
  echo "test"
  LOG=log_eval.txt
  #./predict.sh
fi


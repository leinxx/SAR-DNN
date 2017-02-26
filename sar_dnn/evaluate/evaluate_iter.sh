arch=cnn/deploy_base_fcn.prototxt
#arch=cnn/.fill_deploy_base_fcn.prototxt
cropsize=45
channels=3
scale=1.0
mask=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/mask_land_free
#mask=/media/diskb/sar_dnn/dataset/beaufort_2010_2011/mask
#basedir=/media/diskb/sar_dnn/dataset/beaufort_2010_2011/batches_hhv_$cropsize
#basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_45
basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_65
weightdir=${basedir}/model_SUM_SPLIT_1.0_L1_NO_SIG
mean_std=${basedir}/mean_std_${scale}.txt
ima=${basedir}/ima_used
train_source=${basedir}/batch/train_source_${scale}.txt
valid_source=${basedir}/batch/valid_source_${scale}.txt
test_source=${basedir}/batch/test_source_${scale}.txt
out=${weightdir}/error.txt

for f in $weightdir/*.caffemodel
do
archfill=.fill.prototxt
cp ${arch} ${archfill}
fancyDelim=$(printf '\001')
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" $archfill
sed -i "s${fancyDelim}\$channels${fancyDelim}${channels}${fancyDelim}g" $archfill

# evaluate using model
cmd=$(echo ./evaluate.py --arch $archfill --weights $f --mean_std $mean_std --train_source $train_source --test_source $test_source --valid_source $valid_source --out $out)

# evaluate using predictions
#cmd=$(echo ./evaluate.py --predict_dir $predict --scale $scale --margin $margin --train_source $train_source --test_source $test_source --valid_source $valid_source  --mask_dir $mask --ima_dir $ima --out $out )

echo $cmd
$cmd 2>&1 | tee -a log_eva.txt
done

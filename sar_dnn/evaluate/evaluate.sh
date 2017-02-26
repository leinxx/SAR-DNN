arch=cnn/deploy_base.prototxt
#arch=cnn/.fill_deploy_base_fcn.prototxt
cropsize=45
channels=2
scale=1.0
mask=/home/lein/sar_dnn/dataset/beaufort_gsl/mask_thick
#mask=/media/diskb/sar_dnn/dataset/beaufort_2010_2011/mask
#basedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_45
#basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_45

basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_65
#predict=${basedir}/predict_base_EM_l1_23000
predict=${basedir}/model_BASE_1.0_L2/predict_1.0_8000_crop_1
predict=${1}
mkdir -p $predict
margin=23
weight=${basedir}/model_BASE_1.0_L2/_iter_50000.caffemodel
mean_std=${basedir}/mean_std_${scale}.txt
ima=${basedir}/ima_used
train_source=${basedir}/batch/train_source_${scale}.txt
valid_source=${basedir}/batch/valid_source_${scale}.txt
test_source=${basedir}/batch/test_source_${scale}.txt
out=${predict}/error.txt

archfill=.fill.prototxt
cp ${arch} ${archfill}
fancyDelim=$(printf '\001')
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" $archfill
sed -i "s${fancyDelim}\$channels${fancyDelim}${channels}${fancyDelim}g" $archfill

# evaluate using model
#cmd=$(echo ./evaluate.py --arch $archfill --weights $weight --mean_std $mean_std --train_source $train_source --test_source $test_source --valid_source $valid_source --out $out)

# evaluate using predictions
cmd=$(echo ./evaluate.py --predict_dir $predict --scale $scale --margin $margin --train_source $train_source --test_source $test_source --valid_source $valid_source  --mask_dir $mask --ima_dir $ima --out $out )

echo $cmd
$cmd

./plot_error_bar.sh ${1}

# script used to predict ice concentation from trained model.

stage=0 # stage 0: use input image of scale 1 
arch=deploy_base_fcn.prototxt
featurename=fc5_zoom
scale=1.0
cropsize=45
predict_cropsize=1
model_iter=50000
method=BASE # BASE, EM, SUM, SUM_SPLIT
loss=L2
basedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_hhv_${cropsize}
maskdir=${basedir}/../mask
imagedir=${basedir}/../hhv_$scale
if [ $stage -eq 0 ] # single scale version
then
  scale=1.0
  channels=3
elif [ $stage -eq 1 ] # first stage
then
  scale=0.25
  channels=3
elif [ $stage -eq 2 ]
then
  scale=1.0
  channels=4
  basedir=${basedir}/batch_stage2
else
  echo "invalid stage"
  exit 1
fi
meanfile=${basedir}/mean_std_${scale}.txt
weights=${basedir}/model_${method}_${scale}_${loss}_fc4_6/_iter_${model_iter}.caffemodel
predictdir=${basedir}/predict_${method}_${scale}_${loss}_${model_iter}_fc4_6_crop_${predict_cropsize}
predict_coarse_dir=${basedir}/../predict_base_0.25_l2_1000_crop_3
mkdir -p ${predictdir}
archfill=.fill_${arch}
cp ${arch} ${archfill}

fancyDelim=$(printf '\001')
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" $archfill
sed -i "s${fancyDelim}\$channels${fancyDelim}${channels}${fancyDelim}g" $archfill

#backup configuration
cp ${archfill} $predictdir
cp ${0} $predictdir

for f in ${basedir}/ima_used/*.txt
do
scene=${f##*/}
scene=${scene%%_1.0*}
hh=${scene}-HH-8by8-mat.tif
hv=${scene}-HV-8by8-mat.tif
ia=${scene}-IA.tif
#coarse=${scene}.tif
image=${imagedir}/${hh},${imagedir}/${hv},${imagedir}/${ia} #,${predict_coarse_dir}/${coarse}
mask=${maskdir}/${scene}-mask.tif
predict=${predictdir}/${scene}.tif
if [ -f ${predict} ]
then
  continue
fi
cmd=$(echo "/home/lein/dev/sar_dnn_deeplab/deeplab-public/build/tools/caffe_predict \
    --model=${archfill} \
    --weights=${weights} \
    --meanfile=${meanfile} \
    --featurename=${featurename} \
    --image=${image} \
    --crop_size=$predict_cropsize \
    --mask=$mask \
    --predict=${predict}")
echo $cmd
$cmd
done


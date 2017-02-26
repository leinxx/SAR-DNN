# the solver used here is a concatanation of the caffe solver and caffe arch file
# a separator is added to indicat the seperation of their contents
# this script first seperate the combined file into two files: solver file and arch
# then the solver and arch files will be configured/filled 
stage=0
archfill=$arc
featurename=fc5_zoom
scale=1.0
predict_cropsize=${1}
model_iter=${2}
basedir=../
maskdir=${basedir}/../mask
imagedir=${basedir}/../hhv_$scale
meanfile=${basedir}/mean_std_${scale}.txt
weights=_iter_${model_iter}.caffemodel
predictdir=predict_${scale}_${model_iter}_crop_${predict_cropsize}
mkdir -p ${predictdir}

channels=$channels

for f in ${basedir}/ima_used/*.txt
do
scene=${f##*/}
scene=${scene%%_1.0*}
hh=${scene}-HH-8by8-mat.tif
hv=${scene}-HV-8by8-mat.tif
ia=${scene}-IA.tif
#coarse=${scene}.tif
if [ ${channels} -eq 3 ]
then
image=${imagedir}/${hh},${imagedir}/${hv},${imagedir}/${ia} #,${predict_coarse_dir}/${coarse}
elif [ ${channels} -eq 2 ]
then
  image=${imagedir}/${hh},${imagedir}/${hv}
fi
mask=${maskdir}/${scene}-mask.tif
predict=${predictdir}/${scene}.tif
caffe_predict=$caffe_predict
if [ -f ${predict} ]
then
  continue
fi
cmd=$(echo "${caffe_predict} \
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


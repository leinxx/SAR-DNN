# the solver used here is a concatanation of the caffe solver and caffe arch file
# a separator is added to indicat the seperation of their contents
# this script first seperate the combined file into two files: solver file and arch
# then the solver and arch files will be configured/filled 
#arch=deploy_3layer_fc_stage_1.prototxt
arch=deploy_3layer_fc_stage2.prototxt
cropsize=45
basedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_stage_1_45
imagedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011/hhv
meanfile=${basedir}/mean_std.txt
trainsource=${batchdir}/source.txt
weights=${basedir}/model_stage2_2/_iter_50000.caffemodel
featurename=fc5
predictdir=${basedir}/predict_stage2
mkdir -p ${predictdir}
archfill=.fill_${arch}
cp ${arch} ${archfill}

fancyDelim=$(printf '\001')
sed -i "s${fancyDelim}\$train_mean${fancyDelim}${meanfile}${fancyDelim}g" $archfill
sed -i "s${fancyDelim}\$crop_size${fancyDelim}${cropsize}${fancyDelim}g" $archfill

for f in ${basedir}/predict/*.tif
do
scene=${f##*/}
scene=${scene:0:8}
echo $scene
hh=${scene}-HH-8by8-mat.tif
hv=${scene}-HV-8by8-mat.tif
predict_rescal=${scene}.tif
image=${imagedir}/${hh},${imagedir}/${hv},${basedir}/predict25_rescale/${predict_rescal}
echo $image
predict=${predictdir}/${scene}.tif
if [ -f ${predict} ]
then
  continue
fi
/home/lein/dev/caffe/build/tools/caffe_predict \
    --model=${archfill} \
    --weights=${weights} \
    --image=${image} \
    --meanfile=${meanfile} \
    --featurename=${featurename} \
    --predict=${predict}
done


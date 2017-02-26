# generate sar ice concentration samples
#basedir=/media/diskb/sar_dnn/dataset/beaufort_2010_2011
stage=0
basedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011
maskdir=${basedir}/mask
sardir=${basedir}/hhv_1.0
image_subfixes=-HH-8by8-mat.tif,-HV-8by8-mat.tif,-IA.tif

imadir=${basedir}/ima
ima_subfix=_ima.txt

label_dir=${basedir}/ima_grid
label_subfix=.tif
window=45
land=11
work_dir=${basedir}/batches_${window}_label_noise_${1}
mkdir -p $work_dir

if [ $stage -eq 0 ] # the base version, one stage training
then
  cmd=$(echo "./gen_sar_sample.py --image_dir ${sardir} \
    --image_subfixes \"${image_subfixes}\" \
    --scale 1 \
    --label_dir ${label_dir} \
    --label_subfix ${label_subfix} \
    --ignore_value ${land} \
    --ima_dir ${imadir} 
    --ima_subfix ${ima_subfix} \
    --patch_size ${window} \
    --mask_dir $maskdir \
    --output_dir ${work_dir} \
    --label_noise_std ${1}")
  echo ${cmd}
  $cmd
elif [ $stage -eq 1 ] # first stage of the two stage training
then
  cmd=$(echo "./gen_sar_sample.py --image_dir ${sardir} \
    --image_subfixes \"${image_subfixes}\" \
    --scale 0.25 \
    --label_dir ${label_dir} \
    --label_subfix ${label_subfix} \
    --ignore_value ${land} \
    --ima_dir ${imadir} 
    --ima_subfix ${ima_subfix} \
    --patch_size ${window} \
    --mask_dir $maskdir \
    --output_dir ${work_dir}")
  echo ${cmd}
  $cmd
else # second stage of the two stage training
  predict_stage1_dir=${work_dir}/predict_base_0.25_l2_1000_crop_3
  predict_stage1_subfix=".tif"
  work_dir_stage2=${work_dir}/batch_stage2
  mkdir -p $work_dir_stage2
  margin=$(bc <<< "${window}*4")
  cmd=$(echo "./gen_sar_sample.py --image_dir ${sardir} \
    --image_subfixes \"${image_subfixes}\" \
    --scale 1 \
    --ima_dir ${imadir} 
    --ima_subfix ${ima_subfix} \
    --patch_size ${window} \
    --coarse_predict_dir ${predict_stage1_dir} \
    --coarse_predict_subfix ${predict_stage1_subfix} \
    --margin $margin \
    --mask_dir $maskdir \
    --output_dir ${work_dir_stage2}")
  echo ${cmd}
  $cmd
fi


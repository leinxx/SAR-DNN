#arch=cnn/.fill_deploy_vgg.prototxt
#arch=cnn/.fill_deploy_3layer_fc.prototxt
arch=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_65/model_BASE_1.0_L2_point/deploy.prototxt
basedir=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_65
#basedir=/home/lein/sar_dnn/dataset/beaufort_2010_2011/batches_45/
meanfile=${basedir}/mean_std_1.0.txt
weights=${basedir}/model_BASE_1.0_L2_point/_iter_8000.caffemodel
batch=${basedir}/batch/20140124_215646_1.0.batch
#batch=${basedir}/batch_stage2/batch/20101006_1.0.batch
cmd=$(echo ./vis_gui.py --batch $batch --arch $arch --weights $weights --meanfile $meanfile)
#cmd=$(echo ./vis_gui.py --batch $batch)
echo $cmd
$cmd

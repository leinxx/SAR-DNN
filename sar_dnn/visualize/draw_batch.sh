
batch=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_fcn_45/batch/20140116_223042_1.0.batch
label=/home/lein/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_fcn_45/label/20140116_223042_1.0.batch
start=0
end=10
ext=data
outdir=draw
mkdir -p $outdir
./draw_batch.py $batch $start $end data $outdir
./draw_batch.py $label $start $end label $outdir

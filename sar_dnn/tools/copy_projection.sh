p=/home/lein/Downloads/gdal-2.0.1/swig/python/samples/gdalcopyproj.py

# the image dir to process
src_dir=/home/lein/Sea_ice/images_gsl2014/8by8_with_gcs
for des_dir in $(find /home/lein/sar_dnn/results_geo/gsl -type d)
do
echo $des_dir
  for f in $des_dir/*.tif
do
  if [ -f $f ]
  then
  day=${f##*/}
  day=${day:0:15}
  echo $day
  $p $src_dir/${day}-HH-8by8-mat.tif $f
  fi
done
done

src_dir=/home/lein/Sea_ice/beaufort_8by8/
for des_dir in $(find /home/lein/sar_dnn/results_geo/beaufort -type d)
do
echo $des_dir
  for f in $des_dir/*.tif
do
  if [ -f $f ]
  then
  day=${f##*/}
  day=${day:0:8}
  echo $day
  $p $src_dir/${day}-HH.tif $f
  fi
done
done

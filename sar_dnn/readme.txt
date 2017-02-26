Code used for thesis Learning to Estimate ice Concentration from SAR images -- Lei Wang 2016

Code Structure

caffe: modified deep learning package used for training and prediction. Check "train/train.sh" for examples of using caffe for training and "train/predict.sh" for prediction

evaluate: code and script used to evaluate the predicted ice concentration. Check "evaluate/evaluate.sh" for usage. It generate a table of error statistics in the form of .tex file which can be directly included in latex.

make_sample: generate samples from images and rasterized image analysis data points. Check "make_sample/gen_sar_sample.sh" for usage. Generate samples are called batches (data batche and label batch files containing image patches and their labels separately). These batch files are read in by caffe datum_layer during training.

solver: contains solver and arch files used for training and prediction. Caffe needs solver file and arch file which define the tranining parameters and the network structure. The .solver file in this folder combines solver file and arch file. When training, .solver file will be split to two files by train.sh and used by caffe

tools: a few useful tools used in data cleaning.

train: example script for training and prediction, train.sh, predict.sh

visualize: tools used for visualizing batch data and intermediate features. visualize/run_vis.sh is a GUI to check intermediate features. 



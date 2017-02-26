#include <glog/logging.h>
#include <gflags/gflags.h>

#include <string>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include <boost/algorithm/string.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using std::numeric_limits;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using namespace std;
DEFINE_string(image, "", "Input images, file pathes of different bands of the same image need to be seperated by comma. example: b1.tif,b2.tif ");
DEFINE_string(model, "", "model path");
DEFINE_string(weights, "", "model deploy architecture file");
DEFINE_string(predict, "", "Predict path");
DEFINE_string(meanfile, "", "mean file");
DEFINE_string(featurename, "", "feature name");
DEFINE_int32(crop_size, 11, "crop the center of the feature");
DEFINE_double(scale, 100, "scale factor of the output");
DEFINE_int32(min, 0, "min value of prediction after scale is 0");
DEFINE_int32(max, 100, "max value of prediction after scale is 100");
DEFINE_double(resize, 0.25, "size scale of the 2,3 channels");
DEFINE_string(mask, "", "mask image, non-zero pixels are masked");
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
template <typename Dtype>
void stats_blob(Blob<Dtype>& blob)  {
  Dtype min_val = numeric_limits<Dtype>::min();
  Dtype max_val = numeric_limits<Dtype>::max();
  Dtype sum = 0;
  for (int i = 0; i != blob.count(); ++i) {
     LOG(ERROR) << blob.cpu_data()[i]; 
  }
}

template <typename T>
std::string NumberToString ( T Number )
{
  std::ostringstream ss;
  ss << Number;
  return ss.str();
}

template <typename Dtype>
void write_blob_to_image(Blob<Dtype>& blob, bool is_label)  {
  const int N = blob.num();
  const int C = blob.channels();
  const int W = blob.width();
  const int H = blob.height();
  const int idx = is_label ? 1 : 0;
  for (int n = 0; n != N; ++n)  {
    for (int c = 0; c != C; ++c)  {
      std::string filename = NumberToString(n) + string("_") + string(NumberToString(idx)) + "_" + string(NumberToString(c)) + string("blob.png");
      cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
      int offset = n * W * C * H + c * W * H;
      for (int i = 0; i != H; ++i)  {
        for (int j = 0; j != W; ++j, ++offset)  {
            mat.at<unsigned char>(i,j) = std::max(0, std::min(int(blob.cpu_data()[offset] * 100), 100) );
        }
      }
      cv::imwrite(filename, mat);
    }
  }
}
template <typename Dtype>
void fill_predict(cv::Mat& predict, const Blob<Dtype>* src, const int roff, const int coff, const float scale, const int min_val, const int max_val) {
  const Dtype* src_data = src->cpu_data();
  for (int r = 0; r != src->height(); ++r)  {
    for (int c = 0; c != src->width(); ++c) {
      float v = src_data[r * src->width() + c] * scale;
      v = v > max_val ? max_val : v;
      v = v < min_val ? min_val : v;
      predict.at<unsigned char>(r + roff, c + coff) = v;
    }
  } 
}

template <typename Dtype>
void fill_predict(cv::Mat& predict, const Blob<Dtype>* src, const int crop_size, const vector<cv::Point> locs, const float scale, const int min_val, const int max_val) {
// fill src data to a roi of predict defined bu roff, coff, rows and cols, start index is the index in the roi
  CHECK_EQ(src->channels(), 1);
  CHECK_LE(locs.size(), src->num());
  int crop_radius = crop_size / 2;
  for (int i = 0; i != locs.size(); ++i) {
    const Dtype* src_data = src->cpu_data(i);
    for (int r = -crop_radius; r != crop_size - crop_radius; ++r)  {
      for (int c = -crop_radius; c != crop_size - crop_radius; ++c) {
        float v = src_data[(r + src->height() / 2) * src->width() + (c + src->width() / 2)] * scale;
        v = v > max_val ? max_val : v;
        v = v < min_val ? min_val : v;
        int predict_val = predict.at<unsigned char>(r + locs[i].y, c + locs[i].x);
        //if (predict_val < 0 ) {
          predict.at<unsigned char>(r + locs[i].y, c + locs[i].x) = v;
        //} else  {
        //  predict.at<unsigned char>(r + locs[i].y, c + locs[i].x) = (v + predict_val) / 2;
        
        //}
      }
    } 
  }
}

void fill_predict(cv::Mat& predict, int roff, int coff, int rows, int cols, int start_index, const float* src, int stride, int n, float scale) {
// fill src data to a roi of predict defined bu roff, coff, rows and cols, start index is the index in the roi
// n : number of src data to fill
  for (int i = 0; i != n; ++i) {
    int r = (start_index + i) / cols + roff;
    int c = (start_index + i) % cols + coff;
    float v = src[i * stride] * scale;
    v = v > 100 ? 100 : v;
    v = v < 0 ? 0 : v;
    predict.at<unsigned char>(r, c) =  v / 2;
  }
}

void copy_from_mat(cv::Mat& image, int roff, int coff, int rows, int cols, float mean, float scale, float* dst)  {
  //CHECK_LE(roff + rows, image.rows);
  //CHECK_LE(coff + cols, image.cols);
  //int rend = std::min(roff + rows, image.rows); 
  //int cend = std::min(coff + cols, image.cols); 
  int rend = roff + rows; 
  int cend = coff + cols; 
  for (int i = roff; i != rend; ++i) {
    for (int j = coff; j != cend; ++j)  {
      if (i < 0 || i >= image.rows || j < 0 || j >= image.cols) {
        dst[(i - roff) * cols + j - coff] = 0;
      } else  {
        dst[(i-roff) * cols + j-coff] = (static_cast<float>(image.at<unsigned char>(i, j)) - mean) * scale;
      }
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK_NE(FLAGS_image, "");
  CHECK_NE(FLAGS_model, "");
  CHECK_NE(FLAGS_predict, "");
  CHECK_NE(FLAGS_weights, "");
  CHECK_NE(FLAGS_featurename, "");
  CHECK_NE(FLAGS_meanfile, "");

  // network initialization
  Caffe::SetDevice(0);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  Net<float> caffe_net(FLAGS_model);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  // split mutiple input image file path
  std::vector<std::string> image_files;
  boost::split(image_files, FLAGS_image, boost::is_any_of(","));
  for (int i = 0; i != image_files.size(); ++i) {
    LOG(ERROR) << "band : " << image_files[i];
  }
  // read image files and get patches
  vector<cv::Mat> image_bands;
  int channels = 0;
  for (int band = 0; band != image_files.size(); ++band) {
    image_bands.push_back(cv::imread(image_files[band], -1));
    if (band != 0 && image_bands[band].rows != image_bands[0].rows)  {
      cv::resize(image_bands[band], image_bands[band], cv::Size(image_bands[0].cols, image_bands[0].rows), 0, 0, cv::INTER_NEAREST);
      // CHECK_EQ(image_bands[0].rows, image_bands[band].rows) << "input images should be the same dimension";
      // CHECK_EQ(image_bands[0].cols, image_bands[band].cols) << "input images should be the same dimension";
    }
    channels += image_bands[band].channels();
  }

  // read mask
  cv::Mat mask;
  bool have_mask = false;
  if (FLAGS_mask != "") {
    mask = cv::imread(FLAGS_mask, -1);
    if (mask.rows != image_bands[0].rows || mask.cols != image_bands[0].cols) {
      cv::resize(mask, mask, cv::Size(image_bands[0].cols, image_bands[0].rows), 0, 0, cv::INTER_NEAREST);
      cv::imwrite("test.tif", mask);
    }
    have_mask = true;
  }
  vector<float> mean;
  vector<float> scale;
  LOG(ERROR) << FLAGS_meanfile;
  std::ifstream meanfile(FLAGS_meanfile.c_str());
  scale.resize(channels);
  mean.resize(channels);
  for (int i = 0; i != channels; ++i)  {
    meanfile >> mean[i] >> scale[i];
    scale[i] = 1.0 / scale[i]; 
    LOG(ERROR) << "channel " << i << 
      " : mean " << mean[i] << 
      " std " << 1. / scale[i];
  }

  vector<Blob<float>* > input = caffe_net.input_blobs();
  CHECK_EQ(input.size(), 1);
  Blob<float>* input_data = input[0];
  const int width = input_data->width();
  const int height = input_data->height();
  CHECK_EQ(channels, input_data->channels());
  const int batch_size = input_data->num();
  const int size = width * height * channels;
  const int W = image_bands[0].cols;
  const int H = image_bands[0].rows;
  const int roff = height/2;
  const int coff = width/2;
  cv::Mat predict = cv::Mat::zeros(H, W, CV_8U);
  int n = 0;
  int current_fill_index = 0;
  vector<cv::Point> locs;
  locs.resize(batch_size);
  const int feature_w = caffe_net.blob_by_name(FLAGS_featurename)->width();
  const int feature_h = caffe_net.blob_by_name(FLAGS_featurename)->height();
  const int feature_c = caffe_net.blob_by_name(FLAGS_featurename)->channels();
  CHECK_EQ(feature_c, 1) << "only one channels is supproted due to the implementation here is using opencv";
  for (int r = 0; r < H-height+1; r += FLAGS_crop_size) {
    for (int c = 0; c < W-width+1; c += FLAGS_crop_size)  {
      // extract patch
      if (have_mask == false || mask.at<unsigned char>(r + height / 2, c + width / 2) == 0) {
        float* data = input_data->mutable_cpu_data() + n * size;
        for (int b = 0; b != channels; ++b, data += width * height)  {
          float f = double(image_bands[b].rows) / double(image_bands[0].rows);
          copy_from_mat(image_bands[b], (r + height / 2) * f - height / 2, (c + width / 2) * f - width / 2, height, width, mean[b], scale[b], data);
        }
        locs[n] = cv::Point(c + width / 2, r + height / 2); // assuming the prediction has the same sampling rate as the data
        ++n;
      }
      if (n == batch_size)  {
        // predict
        caffe_net.ForwardPrefilled(); 
        const shared_ptr<Blob<float> > feature_blob = caffe_net.blob_by_name(FLAGS_featurename);
        // write_blob_to_image(*feature_blob.get(), true);
        fill_predict(predict, feature_blob.get(), FLAGS_crop_size, locs, FLAGS_scale, FLAGS_min, FLAGS_max);
        current_fill_index += batch_size;
        n = 0;
        cv::imwrite(FLAGS_predict, predict);
        //caffe_set()
      } 
    }
  }
  if (n) {
    // predict
    caffe_net.ForwardPrefilled(); 
    // use the first n results
    const shared_ptr<Blob<float> > feature_blob = caffe_net.blob_by_name(FLAGS_featurename);
    locs.resize(n);
    fill_predict(predict, feature_blob.get(), FLAGS_crop_size, locs, FLAGS_scale, FLAGS_min, FLAGS_max);
  }
  LOG(ERROR) << "done";
  cv::imwrite(FLAGS_predict, predict);
}


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
DatumDataLayer<Dtype>::~DatumDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void DatumDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  binary_labels_.clear();
  for (int i = 0; i != this->layer_param_.datum_data_param().binary_label_size(); ++i) {
    int b = this->layer_param_.datum_data_param().binary_label(i);
    CHECK_GE(b, 0) << "binary_label is less than 0";
    CHECK_LE(b, 11) << "binary_label is larger than 10";
    binary_labels_.push_back(b);
  }
  // Read the file with filenames
  const string& source = this->layer_param_.datum_data_param().source();
  CHECK_GT(source.size(), 0);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good())
      << "Could not open datum file list (filename: \""+ source + "\")";
  //string folder = source.substr(0, source.rfind('/') + 1);
  string folder = this->layer_param_.datum_data_param().source_dir();
  string filename;
  while (infile >> filename) {
    lines_.push_back(folder + string("/") + filename);
  }
  infile.close();
  target_lines_.clear();
  if (this->layer_param_.datum_data_param().has_target_source())  {
    const string& target_source = this->layer_param_.datum_data_param().target_source();
    CHECK_GT(target_source.size(), 0);
    LOG(INFO) << "Opening file " << target_source;
    std::ifstream target_infile(target_source.c_str());
    CHECK(target_infile.good())
      << "Could not open datum file list (filename: \""+ source + "\")";
    /*
    string folder = target_source.substr(0, source.rfind('/') + 1);
    string filename;
    while (target_infile >> filename) {
      target_lines_.push_back(folder + filename);
    }
    */
    for (int i = 0; i != lines_.size(); ++i)  {
      string fname = lines_[i].substr(lines_[i].rfind('/') + 1, -1);
      fname = target_source + string("/") + fname;
      target_lines_.push_back(fname);
    }

    // validation of two source files
    CHECK_EQ(lines_.size(), target_lines_.size());
    for (int i = 0; i != lines_.size(); ++i)  {
      std::string date1 = lines_[i].substr(lines_[i].rfind('/') + 1, -1);
      std::string date2 = target_lines_[i].substr(target_lines_[i].rfind('/') + 1, -1);
      CHECK_EQ(date1.compare(date2), 0);
    }
  }
  
  lines_idx_.resize(lines_.size());
  for (int i = 0; i != lines_idx_.size(); ++i) {
    lines_idx_[i] = i;
  }

  if (this->layer_param_.datum_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  CHECK(!lines_.empty())
      << "Image list is empty (filename: \"" + source + "\")";
  // Read a data point, and use it to initialize the top blob.
  current_idx_ = 0;
  update_prefetch_buffer();
  Datum datum = prefetch_buffer_[0];
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.datum_data_param().batch_size();
  vector<int> top_shape(4);
  top_shape[0] = batch_size;
  top_shape[1] = datum.channels();
  top_shape[2] = crop_size > 0 ? crop_size : datum.height();
  top_shape[3] = crop_size > 0 ? crop_size : datum.width();
  this->transformed_data_.Reshape(1, top_shape[1], top_shape[2], top_shape[3]);
  this->prefetch_data_.Reshape(batch_size, top_shape[1], top_shape[2], top_shape[3]);
  
  //for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
  //  this->prefetch_[i].data_.Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
  //}
  top[0]->Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (target_lines_.size()) {
    CHECK(this->layer_param_.transform_param().has_label_crop_size());
    int label_crop_size = this->layer_param_.transform_param().label_crop_size();
    int resize_label = this->layer_param_.transform_param().resize_label();
    top_shape[0] = batch_size;
    top_shape[1] = 1;
    top_shape[2] = resize_label < 1 ? label_crop_size : resize_label;
    top_shape[3] = resize_label < 1 ? label_crop_size : resize_label;
  } else {
    top_shape[0] = batch_size;
    top_shape[1] = 1;
    top_shape[2] = 1;
    top_shape[3] = 1;
  }
  top[1]->Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]); 
  this->transformed_label_.Reshape(1, top_shape[1], top_shape[2], top_shape[3]);
  this->prefetch_label_.Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
  //for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
  //  this->prefetch_[i].label_.Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
  //}
  // datum size
}

template <typename Dtype>
void DatumDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_idx_.begin(), lines_idx_.end(), prefetch_rng);
}

template <typename Dtype>
void DatumDataLayer<Dtype>::update_prefetch_buffer() {
  // read two files and shuffle, so each batch have the chance to contain patches from multiple images
  prefetch_buffer_.clear();
  prefetch_target_buffer_.clear();
  int nsample(0), ntarget_sample(0);
  LOG(ERROR) << "load all the data because we can";
  for (int i = 0; i < lines_.size(); ++i) {
    if (lines_id_ >= lines_.size())  {
      ShuffleImages();
      lines_id_ = 0;
    }
    DatumVector buffer;
    std::string current_line = lines_[ lines_idx_[lines_id_] ];
    ReadProtoFromBinaryFileOrDie(current_line, &buffer);
    LOG(ERROR) << current_line << " : " << lines_id_ << ", N samples: " << buffer.datums_size(); 
    nsample += buffer.datums_size();
    for (int idx = 0; idx != buffer.datums_size(); ++idx) {
      prefetch_buffer_.push_back(buffer.datums(idx));
    }
    
    if (target_lines_.size()) {
      std::string current_target_line = target_lines_[lines_idx_[lines_id_]];
      ReadProtoFromBinaryFileOrDie(current_target_line, &buffer);
      LOG(ERROR) << "target: " << current_target_line << " : " << lines_id_ << ", N samples: " << buffer.datums_size(); 
      ntarget_sample += buffer.datums_size();
      CHECK_EQ(nsample, ntarget_sample);
      for (int idx = 0; idx != buffer.datums_size(); ++idx) {
        prefetch_target_buffer_.push_back(buffer.datums(idx));
      }
    }
    lines_id_++;
  }
  LOG(ERROR) << "N samples total : " << nsample;
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //  shuffle(prefetch_buffer_.begin(), prefetch_buffer_.end(), prefetch_rng);
  prefetch_idx_.resize(nsample);
  for (int i = 0; i != nsample; ++i)  {
    prefetch_idx_[i] = i;
  }
  shuffle(prefetch_idx_.begin(), prefetch_idx_.end(), prefetch_rng);
  current_idx_ = 0;
  
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DatumDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->transformed_data_.count());
  Datum datum, label;
  DatumDataParameter datum_data_param = this->layer_param_.datum_data_param();
  const int batch_size = datum_data_param.batch_size();
  const Dtype label_scale = datum_data_param.label_scale();
  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  int nclass = binary_labels_.size(); 
  caffe_set(this->prefetch_label_.count(), Dtype(0.), prefetch_label);
  // datum scales
  for (int idx = 0; idx != batch_size; ++idx) {
    if (current_idx_ == prefetch_buffer_.size() || prefetch_buffer_.size() == 0) {
      // update_prefetch_buffer();
      caffe::rng_t* prefetch_rng =
        static_cast<caffe::rng_t*>(prefetch_rng_->generator());
      //shuffle(prefetch_buffer_.begin(), prefetch_buffer_.end(), prefetch_rng);
      shuffle(prefetch_idx_.begin(), prefetch_idx_.end(), prefetch_rng);
      current_idx_ = 0;
      // LOG(ERROR) << "update_bufer_size: " << prefetch_buffer_.size();
    }
    datum = prefetch_buffer_[ prefetch_idx_[current_idx_] ];
    // write_datum_to_image(datum, idx);
    // Apply transformations (mirror, crop...) to the data
    int offset = this->prefetch_data_.offset(idx);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    if (target_lines_.size()) {
      label = prefetch_target_buffer_[ prefetch_idx_[current_idx_] ];
      this->transformed_label_.set_cpu_data(prefetch_label + this->prefetch_label_.offset(idx));
      this->data_transformer_.Transform(datum, &(this->transformed_data_), &label, &(this->transformed_label_));
    } else {
      this->data_transformer_.Transform(datum, &(this->transformed_data_));
      prefetch_label[idx] = datum.label();
    }

    current_idx_++;

    std::vector<unsigned int>::iterator it;
    it = std::find(binary_labels_.begin(), binary_labels_.end(), datum.label());
    offset = this->prefetch_label_.offset(idx);
    for (int i = 0; i != this->prefetch_label_.width() * this->prefetch_label_.height(); ++i)  {
      if (nclass) {
        for (int j = 0; j != binary_labels_.size(); ++j)  {
          if (prefetch_label[offset + i] > binary_labels_[j] - 1e-2 &
          prefetch_label[offset + i] < binary_labels_[j] + 1e-2)  {
            prefetch_label[offset + i] = j;
          } 
        }
      } else {
        prefetch_label[offset + i] *= label_scale; 
      }
    }
  }
  //this->prefetch_data_.print_stats();
  //this->prefetch_label_.print_stats();

  //write_blob_to_image(prefetch_data_, prefetch_label_);
  // write_blob_to_image(this->prefetch_data_, false);
  // write_blob_to_image(this->prefetch_label_, true);
}

template <typename Dtype>
void DatumDataLayer<Dtype>::write_blob_to_image(Blob<Dtype>& blob, bool is_label)  {
  const int N = blob.num();
  const int C = blob.channels();
  const int W = blob.width();
  const int H = blob.height();
  const int idx = is_label ? 1 : 0;
  for (int n = 0; n != N; ++n)  {
    for (int c = 0; c != C; ++c)  {
      std::string filename = NumberToString(n) + string("_") + string(NumberToString(idx)) + "_" + string(NumberToString(c)) + string("blob.png");
      vector<Dtype> mean = this->data_transformer_.mean_values();
      vector<Dtype> scale= this->data_transformer_.scale_values();
      cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
      int offset = n * W * C * H + c * W * H;
      for (int i = 0; i != H; ++i)  {
        for (int j = 0; j != W; ++j, ++offset)  {
          if (!is_label)  {
            mat.at<unsigned char>(i,j) = blob.cpu_data()[offset] / scale[c] + mean[c] ;
          } else  {
            mat.at<unsigned char>(i,j) = blob.cpu_data()[offset] * 100;
          }
        }
      }
      cv::imwrite(filename, mat);
    }
  }
}

template <typename Dtype>
void DatumDataLayer<Dtype>::write_blob_to_image(Blob<Dtype>& blob, Blob<Dtype>& label)  {
  const int N = blob.num();
  const int C = blob.channels();
  const int W = blob.width();
  const int H = blob.height();
  vector<Dtype> mean = this->data_transformer_.mean_values();
  vector<Dtype> scale= this->data_transformer_.scale_values();
  for (int n = 0; n != N; ++n)  {
    for (int c = 0; c != C; ++c)  {
      std::string filename = NumberToString(n) + string("_") + string(NumberToString(c)) + string("_") + string(NumberToString(label.cpu_data()[n])) + string("blob.png");
      cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
      int offset = n * W * C * H + c * W * H;
      for (int i = 0; i != H; ++i)  {
        for (int j = 0; j != W; ++j, ++offset)  {
          mat.at<unsigned char>(i,j) = blob.cpu_data()[offset] / scale[c] + mean[c] ;
        }
      }
      cv::imwrite(filename, mat);
    }
  }
}
template <typename Dtype>
void DatumDataLayer<Dtype>::write_datum_to_image(Datum& datum, int idx)  {
  const int C = datum.channels();
  const int W = datum.width();
  const int H = datum.height();
  for (int c = 0; c != C; ++c)  {
    std::string filename = string(NumberToString(idx)) + string("_") + string(NumberToString(c)) + string("datum.png");
    cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
    int offset =  c * W * H;
    for (int i = 0; i != H; ++i)  {
      for (int j = 0; j != W; ++j, ++offset)  {
        mat.at<unsigned char>(i,j) = static_cast<uint8_t>(datum.data()[offset]);
      }
    }
    cv::imwrite(filename, mat);
  }
}
INSTANTIATE_CLASS(DatumDataLayer);
REGISTER_LAYER_CLASS(DATUM_DATA, DatumDataLayer);

}  // namespace caffe

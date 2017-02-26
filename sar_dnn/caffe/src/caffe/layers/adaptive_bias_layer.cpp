#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::vector;
using std::sort;
using std::nth_element;
using std::binary_search;
using std::random_shuffle;
using std::numeric_limits;

template <typename Dtype>
void AdaptiveBiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 3) << "bottom: image, score, image analysis";
  AdaptiveBiasParameter param = this->layer_param_.adaptive_bias_param();
  num_iter_ = param.num_iter();
  CHECK_GT(num_iter_, 0);
  select_portion_ = param.select_portion();
  bias_scale_ = param.bias_scale();
  bias_base_ = param.bias_base();
  CHECK(select_portion_ >= 0 && select_portion_ <= 1) << "Select portion needs to be in [0, 1]";
  if (this->layer_param_.adaptive_bias_param().has_mask_rest() )  {
    mask_rest_value_ = this->layer_param_.adaptive_bias_param().mask_rest();
    mask_rest_ = true;
  } else  {
    mask_rest_ = false;
  }
}

template <typename Dtype>
void AdaptiveBiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[1]->num();
  channels_ = bottom[1]->channels();
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();
  //
  CHECK_EQ(bottom[2]->num(), num_) << "Input channels incompatible in num";
  max_labels_ = bottom[2]->channels();
  CHECK_GE(max_labels_, 1) << "Label blob needs to be non-empty";
  CHECK_EQ(bottom[2]->height(), bottom[1]->height()) << "Label height";
  CHECK_EQ(bottom[2]->width(), bottom[1]->width()) << "Label width";
  //
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void AdaptiveBiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0]: image
  // bottom[1]: score
  // bottom[2]: weak score / image analysis 
  // copy bottom[0] -> top[0]
  caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(), top[0]->mutable_cpu_data());
  const int spatial_dim = height_ * width_;
  Blob<Dtype> ylabel(1, 1, height_, width_);
  for (int n = 0; n < num_; ++n) {
    Dtype *top_data = top[0]->mutable_cpu_data(n);
    const Dtype* x = bottom[0]->cpu_data(n);
    // ground truth y_t from bottom[1]
    const Dtype* y_t = bottom[2]->cpu_data(n);
    // score y from bottom[0]
    Dtype* y = ylabel.mutable_cpu_data();
    for (int i = 0; i != spatial_dim; ++i)  {
      y[i] = std::max(Dtype(0), std::min(Dtype(1), bottom[1]->cpu_data(n)[i]));
    }
    // 1. split locations to get working sets S based on y_t, y_t = 0 or 1 are discarded. 
    vector<vector<Dtype> > sets(11); // 0 0.1 0.2 0.3 ... 0.9 1
    vector<Dtype> sets_avg(11, 0); // avg value of each set
    vector<vector<Dtype> > sets_x_std(11); // avg value of each set
    vector<vector<Dtype> > sets_x_avg(11); // avg value of each set
    vector<Dtype> sets_x_std_total(11, Dtype(0) ); // sum of std of all bands of x 
    for (int i = 0; i != sets_x_std.size(); ++i)  {
      sets_x_std[i].assign(bottom[0]->channels(), 0);
      sets_x_avg[i].assign(bottom[0]->channels(), 0); 
    }
    for (int i = 0; i != spatial_dim; ++i ) {
      if (y_t[i] > 1 + 1e-5)  continue;
      int iset = int(y_t[i] * 10 + 0.1);
      sets[iset].push_back(std::max(Dtype(0), std::min(Dtype(1.0), y[i]) ));
      for (int c = 0; c != bottom[0]->channels(); ++c)  {
        sets_x_std[iset][c] += x[i + c * spatial_dim] * x[i + c * spatial_dim];
        sets_x_avg[iset][c] += x[i + c * spatial_dim];
      }
    }

    for (int i = 0; i != sets.size(); ++i)  {
      if (sets[i].size() == 0) continue;
      double sum = std::accumulate(sets[i].begin(), sets[i].end(), 0.0);
      double mean = sum / sets[i].size();
      sets_avg[i] = mean;
      for (int c = 0; c != sets_x_avg[i].size(); ++c)  {
        sets_x_avg[i][c] /= sets[i].size();
        sets_x_std[i][c] = 1e-5 + sets_x_std[i][c] / sets[i].size() - sets_x_avg[i][c] * sets_x_avg[i][c];
        // if (sets_x_std[i][c] < 0) {
        //  LOG(ERROR);
        // }
        sets_x_std_total[i] += sets_x_std[i][c];
        // LOG(ERROR) << sets_x_std[i][c];
      }
      // LOG(ERROR) << sets_x_std_total[i];
      sets_x_std_total[i] = std::sqrt(sets_x_std_total[i]);
      // double sq_sum = std::inner_product(sets[i].begin(), sets[i].end(), sets[i].begin(), 0.0);
      // double stdev = std::sqrt(sq_sum / sets[i].size() - mean * mean);
      // sets_std[i] = stdev;
    }
    // 2. Sort score y in each set S, get target pixels to be supressed or boosted
    // 3. Claculate supression or boost value delta
    // 4. Apply modification
    for (int i = 1; i != sets.size() - 1; ++i) {
      if (sets[i].size() == 0)  {
        continue; 
      }
      int nth = (1 - float(i)/10) * sets[i].size();
      int nselect_low = select_portion_ * nth;
      int nselect_high = sets[i].size() - nth - select_portion_ * (sets[i].size() - nth);
      CHECK_LT(nth, sets[i].size());
      CHECK_LT(nselect_low, sets[i].size());
      CHECK_LT(nselect_high + nth, sets[i].size());
      nth_element(sets[i].begin(), sets[i].begin() + nth, sets[i].end());
      nth_element(sets[i].begin(), sets[i].begin() + nselect_low, sets[i].begin() + nth);
      nth_element(sets[i].begin() + nth, sets[i].begin() + nth + nselect_high, sets[i].end());
      float th_low = sets[i][nselect_low];
      float th_high = sets[i][nselect_high + nth];
      float sum = 0;
      for (int m = 0; m != spatial_dim; ++m)  {
        int iset = int(y_t[m] * 10 + 0.1);
        if (iset == i)  {
          if (y[m]  < th_low) {
            float delta = (fabs(y[m] - sets_avg[i]) + bias_base_) * bias_scale_ * sets_x_std_total[i];
            top_data[m] -= delta;
            if (top_data[m] < 0)  top_data[m] = 0;
          } else if (y[m] > th_high)  {
            float delta = (fabs(y[m] - sets_avg[i]) + bias_base_) * bias_scale_ * sets_x_std_total[i];
            top_data[m] += delta;
            if (top_data[m] > 1) top_data[m] = 1;
          }           
          sum += top_data[m];
          if (mask_rest_ && y[m] >= th_low && y[m] <= th_high) {
            top_data[m] = mask_rest_value_; // mask the rest  
          }

        }
      }
      float mean_delta = sum / sets[i].size() - float(i) / 10;
      // keep mean
      for (int m = 0; m != spatial_dim; ++m)  {
        int iset = int(y_t[m] * 10 + 0.1);
        if (iset == i)  {
          top_data[m] -= mean_delta;
        }
      }
    }
  }
  /*
  for(int s = 0; s != bottom[0]->count(); ++s)  {
    bottom[0]->mutable_cpu_data()[s] = bottom[0]->cpu_data()[s] * 33 + 75;
  }
  write_blob_to_uint8_image("data.png", bottom[0], -1, -1, Dtype(1), 0, 255);
  write_blob_to_uint8_image("score.png", bottom[1], -1, -1, Dtype(100), 0, 100);
  write_blob_to_uint8_image("before.png", bottom[2], -1, -1, Dtype(100), 0, 110);
  write_blob_to_uint8_image("after.png", top[0], -1, -1, Dtype(100), 0, 110);
  */
}

template <typename Dtype>
void AdaptiveBiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
//STUB_GPU(AdaptiveBiasLayer);
#endif

INSTANTIATE_CLASS(AdaptiveBiasLayer);
REGISTER_LAYER_CLASS(ADAPTIVE_BIAS, AdaptiveBiasLayer);

}  // namespace caffe

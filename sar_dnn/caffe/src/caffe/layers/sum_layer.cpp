#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {
template <typename Dtype>
bool in_vec(const vector<Dtype>& vec, const Dtype val) {
  for (int i = 0; i != vec.size(); ++i) {
    if (fabs(vec[i] - val) < 1e-5) {
      return true;
    }
  }
  return false;
}

template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ignore_label_.clear();
  for (int i = 0; i != this->layer_param().sum_param().ignore_label_size(); ++i)  {
    ignore_label_.push_back(this->layer_param().sum_param().ignore_label(i));
  }
  split_sum_ = this->layer_param().sum_param().split_sum(); 
}

template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (split_sum_) {
    top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
    top[1]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  } else {
    top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
    top[1]->Reshape(bottom[0]->num(), 1, 1, 1);
  }
}


template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // label wise sum for all the pixels except the ignored labels. 
  
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  count_ = vector<vector<int> >(bottom[0]->num());
  for (int n = 0; n != bottom[0]->num(); ++n) {
    count_[n].resize(12, 0); // count for the 12 levels of ice concentration
  }
  int n_pixels =  bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();
  if (split_sum_) {
    for (int n = 0; n != bottom[0]->num(); ++n) {
      Dtype data_sum[12] = {};
      Dtype* top_data = top[0]->mutable_cpu_data(n);
      Dtype* top_label = top[1]->mutable_cpu_data(n);
      const Dtype* data = bottom[0]->cpu_data(n);
      const Dtype* label = bottom[1]->cpu_data(n);
      for (int i = 0; i !=  n_pixels; ++i) {
        // assume label is between 0 and 1
        int label_index = int(label[i] * 10 + 1e-5);
        count_[n][label_index]++;
        data_sum[label_index] += data[i];
      }
      for (int i = 0; i != 12; ++i) {
        if (count_[n][i]) {
          data_sum[i] /= count_[n][i];
        } else  {
          data_sum[i] = Dtype(i) / 10.0;
        }
        //top_label[i] = Dtype(i) / 10.0;
      }
      for (int i = 0; i != n_pixels; ++i) {
        if (!in_vec(ignore_label_, label[i])) {
          top_data[i] = data_sum[int(label[i] * 10 + 1e-5)];
        } else  {
          top_data[i] = data[i];
        }
        top_label[i] = label[i];
      }
    }
  } else  {
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* top_label = top[1]->mutable_cpu_data();
    for (int n = 0; n != bottom[0]->num(); ++n) {
      const Dtype* data = bottom[0]->cpu_data(n);
      const Dtype* label = bottom[1]->cpu_data(n);
      for (int i = 0; i !=  n_pixels; ++i) {
        if (!in_vec(ignore_label_, label[i])) {
          count_[n][0]++;
          top_data[n] += data[i];
          top_label[n] += label[i];
        }
      }
      top_data[n] /= count_[n][0];
      top_label[n] /= count_[n][0];
    }
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(propagate_down[1], false);
  int n_pixels =  bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();
  if (split_sum_) {
    for (int n = 0; n != bottom[0]->num(); ++n) {
      Dtype* top_data_diff = top[0]->mutable_cpu_diff(n);
      Dtype* top_label_diff = top[1]->mutable_cpu_diff(n);
      Dtype* diff_data = bottom[0]->mutable_cpu_diff(n);
      Dtype* diff_label = bottom[1]->mutable_cpu_diff(n);
      const Dtype* label = bottom[1]->cpu_data(n);
      Dtype diff_split[12] = {};
      for (int i = 0; i != n_pixels; ++i) {
        int label_index = int(label[i] * 10 + 1e-5);
        diff_split[label_index] += top_data_diff[i] / count_[n][label_index];
        diff_label[i] = 0;
      }
      for (int i = 0; i != n_pixels; ++i) {
        int label_index = int(label[i] * 10 + 1e-5);
        if (in_vec(ignore_label_, label[i])) {
          diff_data[i] = top_data_diff[i];
        } else {
          diff_data[i] = diff_split[label_index];
        }
      }
    }
  } else  {
    Dtype* top_data_diff = top[0]->mutable_cpu_diff();
    Dtype* top_label_diff = top[1]->mutable_cpu_diff();
    for (int n = 0; n != bottom[0]->num(); ++n) {
      Dtype* diff_data = bottom[0]->mutable_cpu_diff(n);
      Dtype* diff_label = bottom[1]->mutable_cpu_diff(n);
      const Dtype* label = bottom[1]->cpu_data(n);
      for (int i = 0; i != n_pixels; ++i) {
        if (!in_vec(ignore_label_, label[i])) {
          diff_data[i] = top_data_diff[n] / count_[n][0]; 
          if (propagate_down[1])  {
            diff_label[i] = top_label_diff[n] / count_[n][0]; 
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(SUM, SumLayer);


bool in_vec(const vector<float>& vec, const float val);
bool in_vec(const vector<double>& vec, const double val);
}  // namespace caffe

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void L1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  ignore_.ReshapeLike(*bottom[0]);
  caffe_set(ignore_.count(), Dtype(0), ignore_.mutable_cpu_data());
  ignore_label_.clear();
  for (int i = 0; i != this->layer_param_.loss_param().ignore_label_size(); ++i)  {
    ignore_label_.push_back(this->layer_param_.loss_param().ignore_label(i));
  }
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* y = bottom[0]->cpu_data();
  const Dtype* y_t = bottom[1]->cpu_data(); 
  Dtype loss = 0;
  count_valid_ = 0;
  for (int i = 0; i != bottom[0]->count(); ++i)  {
    bool ignore = false;
    for (int j = 0; j != ignore_label_.size(); ++j)  {
      if (fabs(y_t[i] - ignore_label_[j]) < 1e-5 ) {
        ignore_.mutable_cpu_data()[i] = true;
        ignore = true;
        break;
      }
    }
    if (ignore == false) {
      loss += fabs(y[i] - y_t[i]);
      ++count_valid_; 
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count_valid_;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      for (int j = 0; j != bottom[0]->count(); ++j)  {
        if (ignore_.cpu_data()[j])  {
          bottom[i]->mutable_cpu_diff()[j] = 0.0;
          continue;
        }
        Dtype sign = bottom[i]->cpu_data()[j] - bottom[(i + 1)  % 2]->cpu_data()[j];
        if (sign > 0 ) {
          bottom[i]->mutable_cpu_diff()[j] = loss_weight / count_valid_;
        } else if (sign == 0) {
          bottom[i]->mutable_cpu_diff()[j] = 0.0;
        } else  {
          bottom[i]->mutable_cpu_diff()[j] = -loss_weight / count_valid_;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS(L1LossLayer);
REGISTER_LAYER_CLASS(L1_LOSS, L1LossLayer);

}  // namespace caffe

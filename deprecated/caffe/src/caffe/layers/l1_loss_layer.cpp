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
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* y = bottom[0]->cpu_data();
  const Dtype* y_t = bottom[1]->cpu_data(); 
  Dtype loss = 0;
  for (int i = 0; i != bottom[0]->count(0); ++i)  {
    loss += fabs(y[i] - y_t[i]);
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->count();
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      for (int j = 0; j != bottom[0]->count(0); ++j)  {
        Dtype sign = bottom[i]->cpu_data()[j] - bottom[(i + 1)  % 2]->cpu_data()[j];
        if (sign > 0 ) {
          bottom[i]->mutable_cpu_diff()[j] = loss_weight / bottom[0]->count();
        } else if (sign == 0) {
          bottom[i]->mutable_cpu_diff()[j] = 0.0;
        } else  {
          bottom[i]->mutable_cpu_diff()[j] = -loss_weight / bottom[0]->count();
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS(L1LossLayer);
REGISTER_LAYER_CLASS(L1Loss);

}  // namespace caffe

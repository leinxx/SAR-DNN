#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConstrainLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ConstrainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* y = bottom[0]->cpu_data();
  const Dtype* y_t = bottom[1]->cpu_data(); 
  const Dtype beta = this->layer_param_.loss_param().beta();
  Dtype loss = 0;
  for (int i = 0; i != bottom[0]->count(0); ++i)  {
    if (y[i] < 0 || y[i] > 1) {
      continue;
    } else  {
      loss += y[i] * ( 1 - y[i]) / ( y_t[i] * (1 - y_t[i]) + beta );
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->count();
}

template <typename Dtype>
void ConstrainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  const Dtype scale = loss_weight / bottom[0]->count();
  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* data_gt = bottom[1]->cpu_data();
  const Dtype beta = this->layer_param_.loss_param().beta();
  Dtype* diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    for (int j = 0; j != bottom[0]->count(0); ++j)  {
      if (data[j] > 1 || data[j] < 0) {
        diff[j] = 0;
      } else  {
        diff[j] = scale * (1 - 2 * data[j]) / (data_gt[j] * (1 - data_gt[j]) + beta); 
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConstrainLossLayer);
#endif

INSTANTIATE_CLASS(ConstrainLossLayer);
REGISTER_LAYER_CLASS(ConstrainLoss);

}  // namespace caffe

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void generate_mask(const int count, const Dtype* data, const Dtype* ignore_value, const int n_ignore_value, Dtype* mask, int& count_valid)  {
    count_valid = 0;
    Dtype EPS = 1e-5;
    CUDA_KERNEL_LOOP(i, count)  {
      bool ignore = false;
      for (int j = 0; j != n_ignore_value; ++j)  {
        if (data[i] > ignore_value[j] - EPS && data[i] < ignore_value[j] + EPS) {
          ignore = true;
          break;
        }
      }
      if (ignore) {
        mask[i] = 0;
      } else  {
        mask[i] = 1;
        ++count_valid;
      }
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<Dtype> ignore_value;
  for (int i = 0; i != this->layer_param_.loss_param().ignore_label_size(); ++i) {
    ignore_value[i] ==  this->layer_param_.loss_param().ignore_label(i);
  }

  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  Blob<Dtype> mask;
  mask.ReshapeLike(*bottom[1]);
  if (ignore_value.size())  {
    generate_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom[1]->gpu_data(), &ignore_value[0], ignore_value.size(), mask.mutable_gpu_data(), count_valid_);
  } else  {
    count_valid_ = count;
    caffe_gpu_set(count, Dtype(1.0), mask.mutable_gpu_data() );
  }
  caffe_gpu_mul(count, diff_.gpu_data(), mask.gpu_data(), diff_.mutable_gpu_data()); 
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  loss_ = 0.5 * dot / count_valid_;
  top[0]->mutable_cpu_data()[0] = loss_;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      Dtype alpha = sign * top[0]->cpu_diff()[0] / count_valid_;
      caffe_gpu_axpby(
          count,                              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe

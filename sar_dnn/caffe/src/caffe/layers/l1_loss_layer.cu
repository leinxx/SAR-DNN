#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void generate_mask_2(const int count, const Dtype* data, const Dtype ignore_value, const int n_ignore_value, Dtype* mask, int& count_valid)  {
    count_valid = 0;
    Dtype EPS = 1e-5;
    CUDA_KERNEL_LOOP(i, count)  {
      bool ignore = false;
      for (int j = 0; j != n_ignore_value; ++j)  {
        if (data[i] > ignore_value - EPS && data[i] < ignore_value + EPS) {
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

template<typename Dtype>
__global__ void sign_kernel(const int n, const Dtype* x, const Dtype a, Dtype* y)  {
  CUDA_KERNEL_LOOP(index, n)  {
    if (x[index] > 0) {
      y[index] = 1 * a;
    } else if (x[index] == 0) {
      y[index] = 0;
    } else  {
      y[index] = -1 * a;
    }
  }
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
  return;
  const int count = bottom[0]->count();
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());
  count_valid_ = 0; 
  Blob<Dtype> mask;
  if (ignore_label_.size())  {
    mask.ReshapeLike(*bottom[0]);
    LOG(ERROR) << ignore_.cpu_data()[0];
    generate_mask_2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom[1]->gpu_data(), ignore_label_[0], ignore_label_.size(), mask.mutable_gpu_data(), count_valid_);
    CUDA_POST_KERNEL_CHECK;
    for (int i = 0; i != count; ++i)  {
      LOG(ERROR) << mask.cpu_data()[i];
    }
  } else  {
    count_valid_ = count;
    caffe_gpu_set(count, Dtype(1), mask.mutable_gpu_data() );
  }
  caffe_gpu_mul(count, diff_.gpu_data(), mask.gpu_data(), diff_.mutable_gpu_data()); 
  Dtype asum = 0;
  caffe_gpu_asum(count, diff_.gpu_data(), &asum);
  Dtype loss = asum / count_valid_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
  return;
  const int N = bottom[0]->count();
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
    sign_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, diff_.gpu_data(), Dtype(1 - 2 * i) * loss_weight / count_valid_, bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe

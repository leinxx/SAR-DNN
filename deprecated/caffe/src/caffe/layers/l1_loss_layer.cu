#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

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
  const int count = bottom[0]->count();
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());
  Dtype asum = 0;
  caffe_gpu_asum(count, diff_.gpu_data(), &asum);
  Dtype loss = asum / bottom[0]->count();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int N = bottom[0]->count();
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
    sign_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, diff_.gpu_data(), Dtype(1 - 2 * i) * loss_weight / bottom[0]->count(), bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe

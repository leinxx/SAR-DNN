#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void binary_constrain_kernel(const int n, const Dtype* x, const Dtype* xt, const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n)  {
    y[index] = x[index] * (1 - x[index]) / (xt[index] * (1 - xt[index]) + beta);
  }
}

template <typename Dtype>
__global__ void binary_constrain_gradient_kernel(const int n, const Dtype* x, const Dtype* xt, const Dtype beta, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n)  {
    y[index] = (1 - 2 * x[index]) / (xt[index] * (1 - xt[index]) + beta);
  }
}

template <typename Dtype>
void ConstrainLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
  return;
  const int count = bottom[0]->count();
  const Dtype beta = this->layer_param_.loss_param().beta();
  binary_constrain_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), beta, diff_.mutable_gpu_data() );
  Dtype asum = 0;
  caffe_gpu_asum(count, diff_.gpu_data(), &asum);
  Dtype loss = asum / bottom[0]->count();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ConstrainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
  return;
  const int N = bottom[0]->count();
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  const Dtype beta = this->layer_param_.loss_param().beta();
  if (propagate_down[0]) {
    binary_constrain_gradient_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, bottom[0]->gpu_data(), bottom[1]->gpu_data(), beta, diff_.mutable_gpu_diff() );
    caffe_gpu_scale(N, loss_weight / N, diff_.mutable_gpu_diff(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConstrainLossLayer);

}  // namespace caffe

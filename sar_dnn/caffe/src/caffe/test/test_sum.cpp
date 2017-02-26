#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SumLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 1, 14, 14)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 14, 14)),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {
    // fill the values
    ignore_value_ = 1.1;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int count = this->blob_bottom_label_->num() * this->blob_bottom_label_->channels() * 
      this->blob_bottom_label_->height() * this->blob_bottom_label_->width();
    caffe_set(count, Dtype(0), this->blob_bottom_label_->mutable_cpu_data());
    for (int i = 0; i != 11; ++i) {
      for(int j = 0; j != 10; ++j)  {
        this->blob_bottom_label_->mutable_cpu_data()[i * 10 + j] = Dtype(j) / 10.0;
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }
  virtual ~SumLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_data_;
    delete blob_top_label_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    layer_param.mutable_sum_param()->add_ignore_label(1.1);
    
    SumLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    int count_1 = this->blob_bottom_label_->channels() * 
      this->blob_bottom_label_->height() * this->blob_bottom_label_->width();
    for (int n = 0; n != blob_bottom_data_->num(); ++n)  {
      Dtype sum_data(0), sum_label(0);
      Dtype count(0);
      for (int i = 0; i != count_1; ++i)  {
        if (fabs(*(blob_bottom_label_->cpu_data(n) + i) - 1.1 ) > 1e-5 )  {
          sum_data += *(blob_bottom_data_->cpu_data(n) + i);
          sum_label += *(blob_bottom_label_->cpu_data(n) + i);
          count++;
        }
      }
      sum_data /= count;
      sum_label /= count;
      Dtype data1 = *blob_top_data_->cpu_data(n);
      Dtype label1 = *blob_top_label_->cpu_data(n);
    EXPECT_NEAR(sum_data, data1, 1e-4);
    EXPECT_NEAR(sum_label, label1, 1e-4);
    }
  }

  void TestForward_split()  {
    
    LayerParameter layer_param;
    layer_param.mutable_sum_param()->add_ignore_label(1.1);
    layer_param.mutable_sum_param()->add_ignore_label(1);
    layer_param.mutable_sum_param()->add_ignore_label(0);
    layer_param.mutable_sum_param()->set_split_sum(true);
    SumLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    int count_1 = this->blob_bottom_label_->channels() * 
      this->blob_bottom_label_->height() * this->blob_bottom_label_->width();
    for (int n = 0; n != blob_bottom_data_->num(); ++n)  {
      for (int j = 0; j != 12; ++j)  {
        int c = 0;
        Dtype sum_data(0), sum_label(0);
        for (int i = 0; i != count_1; ++i)  {
          if (fabs(*(blob_bottom_label_->cpu_data(n) + i) - Dtype(j) / 10.0 ) < 1e-5 )  {
            if (j == 0 || j == 11 || j == 10) {
              EXPECT_NEAR(blob_top_data_->cpu_data(n)[i], blob_bottom_data_->cpu_data(n)[i], 1e-5);
            }
            sum_data += blob_bottom_data_->cpu_data(n)[i]; 
            c++;
          }
        }
        if (c) {
          sum_data /= c;
        } else {
          sum_data = Dtype(j) / 10.0;
        }
        if (j != 0 && j != 10 && j != 11) {
          for (int i = 0; i != count_1; ++i)  {
            if (fabs(blob_bottom_label_->cpu_data(n)[i] - Dtype(j) / 10.0) < 1e-5) {
              EXPECT_NEAR(sum_data, blob_top_data_->cpu_data(n)[i], 1e-5);
              EXPECT_NEAR(blob_bottom_label_->cpu_data(n)[i], blob_top_label_->cpu_data(n)[i], 1e-5);
            }
          }
        }
      }
    }
  }

  int ignore_value_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SumLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SumLayerTest, TestForward_split) {
 this->TestForward_split(); 
}

TYPED_TEST(SumLayerTest, TestGradientL2_split) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_sum_param()->add_ignore_label(1.1);
  layer_param.mutable_sum_param()->add_ignore_label(1);
  layer_param.mutable_sum_param()->add_ignore_label(0);
  layer_param.mutable_sum_param()->set_split_sum(true);
  SumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
TYPED_TEST(SumLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_sum_param()->add_ignore_label(1.1);
  SumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe

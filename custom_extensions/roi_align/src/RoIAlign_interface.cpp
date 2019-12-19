/* adopted from pytorch framework
    https://github.com/pytorch/vision/blob/master/torchvision/csrc/ROIAlign.h on Nov 15 2019.

    does not include CPU support but could be added with this interface.
*/

#include <torch/extension.h>

// Declarations that are initialized in cuda file
at::Tensor ROIAlign_forward_cuda(const at::Tensor& input, const at::Tensor& rois, const float spatial_scale,
                                const int pooled_height, const int pooled_width, const int sampling_ratio);
at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad, const at::Tensor& rois, const float spatial_scale,
    const int pooled_height, const int pooled_width, const int batch_size, const int channels,
    const int height, const int width, const int sampling_ratio);

// Interface for Python
at::Tensor ROIAlign_forward(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be  scaled to this.
    const int64_t pooled_height, // The height of the pooled feature map.
    const int64_t pooled_width, // The width of the pooled feature
    const int64_t sampling_ratio) // The number of points to sample in each bin along each axis.
{
  if (input.type().is_cuda()) {
    return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  }
  AT_ERROR("Not compiled with CPU support");
  //return ROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad, const at::Tensor& rois, const float spatial_scale,
    const int pooled_height, const int pooled_width, const int batch_size, const int channels,
    const int height, const int width, const int sampling_ratio) {
  if (grad.type().is_cuda()) {
    return ROIAlign_backward_cuda( grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels,
        height, width, sampling_ratio);
  }
  AT_ERROR("Not compiled with CPU support");
  //return ROIAlign_backward_cpu(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels,
  //   height, width, sampling_ratio);
}

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width,
      const int64_t sampling_ratio) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["input_shape"] = input.sizes();
    ctx->save_for_backward({rois});
    auto result = ROIAlign_forward(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
    return {result};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = ROIAlign_backward(
        grad_output[0],
        rois,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        input_shape[0], //b
        input_shape[1], //c
        input_shape[2], //h
        input_shape[3], //w
        ctx->saved_data["sampling_ratio"].toInt());
    return {
        grad_in, Variable(), Variable(), Variable(), Variable(), Variable()};
  }
};

Tensor roi_align(const Tensor& input, const Tensor& rois, const double spatial_scale, const int64_t pooled_height,
    const int64_t pooled_width, const int64_t sampling_ratio) {

  return ROIAlignFunction::apply(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)[0];

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align", &roi_align, "ROIAlign 2D in c++ and/or cuda");
}
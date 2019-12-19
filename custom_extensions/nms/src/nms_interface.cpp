/*  adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/csrc/nms.h on Nov 15 2019
    no cpu support, but could be added with this interface.
*/
#include <torch/extension.h>


//#include "cpu/vision_cpu.h"

at::Tensor nms_cuda(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold);

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  if (dets.device().is_cuda()) {

    if (dets.numel() == 0) {
      //at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);

  }
  AT_ERROR("Not compiled with CPU support");
  //at::Tensor result = nms_cpu(dets, scores, iou_threshold);
  //return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "NMS C++ and/or CUDA");
}
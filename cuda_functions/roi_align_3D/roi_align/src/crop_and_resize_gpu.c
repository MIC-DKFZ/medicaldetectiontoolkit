#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"

extern THCState *state;


void crop_and_resize_gpu_forward(
    THCudaTensor * image,
    THCudaTensor * boxes,           // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    const int crop_zdepth,
    THCudaTensor * crops
) {
    const int batch_size = THCudaTensor_size(state, image, 0);
    const int depth = THCudaTensor_size(state, image, 1);
    const int image_height = THCudaTensor_size(state, image, 2);
    const int image_width = THCudaTensor_size(state, image, 3);
    const int image_zdepth = THCudaTensor_size(state, image, 4);

    const int num_boxes = THCudaTensor_size(state, boxes, 0);

    // init output space
    THCudaTensor_resize5d(state, crops, num_boxes, depth, crop_height, crop_width, crop_zdepth);
    THCudaTensor_zero(state, crops);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        THCudaTensor_data(state, image),
        THCudaTensor_data(state, boxes),
        THCudaIntTensor_data(state, box_index),
        num_boxes, batch_size, image_height, image_width, image_zdepth,
        crop_height, crop_width, crop_zdepth, depth, extrapolation_value,
        THCudaTensor_data(state, crops),
        stream
    );
}


void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    THCudaTensor * boxes,      // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
) {
    // shape
    const int batch_size = THCudaTensor_size(state, grads_image, 0);
    const int depth = THCudaTensor_size(state, grads_image, 1);
    const int image_height = THCudaTensor_size(state, grads_image, 2);
    const int image_width = THCudaTensor_size(state, grads_image, 3);
    const int image_zdepth = THCudaTensor_size(state, grads_image, 4);

    const int num_boxes = THCudaTensor_size(state, grads, 0);
    const int crop_height = THCudaTensor_size(state, grads, 2);
    const int crop_width = THCudaTensor_size(state, grads, 3);
    const int crop_zdepth = THCudaTensor_size(state, grads, 4);

    // init output space
    THCudaTensor_zero(state, grads_image);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        THCudaTensor_data(state, grads),
        THCudaTensor_data(state, boxes),
        THCudaIntTensor_data(state, box_index),
        num_boxes, batch_size, image_height, image_width, image_zdepth,
        crop_height, crop_width, crop_zdepth, depth,
        THCudaTensor_data(state, grads_image),
        stream
    );
}
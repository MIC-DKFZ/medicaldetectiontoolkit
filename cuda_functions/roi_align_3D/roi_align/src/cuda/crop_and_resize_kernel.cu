#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_zdepth, int crop_height, int crop_width, int crop_zdepth, int depth,
    float extrapolation_value, float *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads) // nthreads = total_count!
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b)) position in out grid!!!
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))   NCYX yes seems like xy is exchanged!
        // NCHWZ: out_idx = z + crop_zdepth * (w + crop_width * (h + crop_height * (d + depth * b))) z == last.

        int idx = out_idx;

        const int z = idx % crop_zdepth;
        idx /= crop_zdepth;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;

        const int d = idx % depth;
        const int b = idx / depth; // batch

        const float y1 = boxes_ptr[b * 6]; // b = batch -> 0 // normalized coords!!
        const float x1 = boxes_ptr[b * 6 + 1];
        const float y2 = boxes_ptr[b * 6 + 2];
        const float x2 = boxes_ptr[b * 6 + 3];
        const float z1 = boxes_ptr[b * 6 + 4];
        const float z2 = boxes_ptr[b * 6 + 5];

        const int b_in = box_ind_ptr[b]; // == 0 in my case.
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        // e.g. (0.4-0.3)*100 = 10 / 7 = 1.3 ratio proposal_size / crops_size. one cell in crops has size 1.3 in_pixel.

        const float height_scale =
            (crop_height > 1) ? (y2 - y1)  * (image_height ) / (crop_height ) : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width ) / (crop_width ) : 0;

        const float zdepth_scale =
            (crop_zdepth > 1) ? (z2 - z1) * (image_zdepth ) / (crop_zdepth ) : 0;


        // e.g.  0.3*100 + 5 * 1.3 . Which floating coordinate is going into cell?
        // e.g. y: 30 (lower bound prop) + 7.5 (current crop position * scale)


        float tmp_in_y = (crop_height > 1)
                                ? y1 * (image_height ) + y * height_scale + height_scale/2 - 0.5
                                : 0.5 * (y1 + y2) * (image_height);

        if (tmp_in_y > image_height - 1)
        {
         tmp_in_y = image_height - 1;
        }
        if (tmp_in_y < 0)
        {
         tmp_in_y = 0;
        }
        const float in_y = tmp_in_y;


        float tmp_in_x = (crop_width > 1)
                                ? x1 * (image_width ) + x * width_scale + width_scale/2 - 0.5
                                : 0.5 * (x1 + x2) * (image_width );

        if (tmp_in_x > image_width - 1)
        {
         tmp_in_x = image_width - 1;
        }
        if (tmp_in_x < 0)
        {
         tmp_in_x= 0;
        }
	    const float in_x = tmp_in_x;


        float tmp_in_z = (crop_zdepth > 1)
                            ? z1 * (image_zdepth ) + z * zdepth_scale + zdepth_scale/2 - 0.5
                            : 0.5 * (z1 + z2) * (image_zdepth);

        if (tmp_in_z > image_zdepth - 1)
        {
         tmp_in_z = image_zdepth - 1;
        }
        if (tmp_in_z < 0)
        {
         tmp_in_z= 0;
        }
        const float in_z = tmp_in_z;

        // this is just rounding of the floating coord of grid cell. The distances to nearest grid points are
        // memorized (lerp) to be used for bilinear interpolation later.
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index; //

        const int front_z_index = floorf(in_z);
        const int back_z_index = ceilf(in_z);
        const float z_lerp = in_z - front_z_index;


        // address of image + going to the right feature map.
        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width * image_zdepth;

        // 1D address of corner points of in_coords to grid cell.
        // NCHWZ: out_idx = z + crop_zdepth * (w + crop_width * (h + crop_height * (d + depth * b))) z == last.
        const float top_left_front = pimage[front_z_index + image_zdepth * (left_x_index + image_width * top_y_index)];
        const float top_right_front = pimage[front_z_index + image_zdepth * (right_x_index + image_width * top_y_index)];
        const float bottom_left_front = pimage[front_z_index + image_zdepth * (left_x_index + image_width * bottom_y_index)];
        const float bottom_right_front = pimage[front_z_index + image_zdepth * (right_x_index + image_width * bottom_y_index)];
        const float top_left_back = pimage[back_z_index + image_zdepth * (left_x_index + image_width * top_y_index)];
        const float top_right_back = pimage[back_z_index + image_zdepth * (right_x_index + image_width * top_y_index)];
        const float bottom_left_back = pimage[back_z_index + image_zdepth * (left_x_index + image_width * bottom_y_index)];
        const float bottom_right_back = pimage[back_z_index + image_zdepth * (right_x_index + image_width * bottom_y_index)];

        // Bilinear Interpolation!! These are pixel values now! lerp is the interpolation distance!
        // No Maxpool, only one point is sampled!
        const float top_front = top_left_front + (top_right_front - top_left_front) * x_lerp;
        const float bottom_front = bottom_left_front + (bottom_right_front - bottom_left_front) * x_lerp;
        const float top_back = top_left_back + (top_right_back - top_left_back) * x_lerp;
        const float bottom_back = bottom_left_back + (bottom_right_back - bottom_left_back) * x_lerp;

        const float front = top_front + (bottom_front - top_front) * y_lerp;
        const float back = top_back + (bottom_back - top_back) * y_lerp;

        crops_ptr[out_idx] = front + (back - front) * z_lerp; // assign interpolated value to Grid cell!


    }
}

__global__
void CropAndResizeBackpropImageKernel(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_zdepth, int crop_height, int crop_width, int crop_zdepth, int depth,
    float *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        // NCHWZ: out_idx = z + crop_zdepth * (w + crop_width * (h + crop_height * (d + depth * b))) z == last.
        int idx = out_idx;

        const int z = idx % crop_zdepth;
        idx /= crop_zdepth;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 6]; // b = batch -> 0 // normalized coords!!
        const float x1 = boxes_ptr[b * 6 + 1];
        const float y2 = boxes_ptr[b * 6 + 2];
        const float x2 = boxes_ptr[b * 6 + 3];
        const float z1 = boxes_ptr[b * 6 + 4];
        const float z2 = boxes_ptr[b * 6 + 5];


        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height ) / (crop_height )
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width ) / (crop_width ) : 0;

        const float zdepth_scale =
            (crop_zdepth > 1) ? (z2 - z1) * (image_zdepth ) / (crop_zdepth ) : 0;


        float tmp_in_y = (crop_height > 1)
                                ? y1 * (image_height ) + y * height_scale + height_scale/2 - 0.5
                                : 0.5 * (y1 + y2) * (image_height);
        if (tmp_in_y > image_height - 1)
        {
         tmp_in_y = image_height - 1;
        }
        if (tmp_in_y < 0)
        {
         tmp_in_y = 0;
        }
        const float in_y = tmp_in_y;


        float tmp_in_x = (crop_width > 1)
                                ? x1 * (image_width ) + x * width_scale + width_scale/2 - 0.5
                                : 0.5 * (x1 + x2) * (image_width );
        if (tmp_in_x > image_width - 1)
        {
         tmp_in_x = image_width - 1;
        }
        if (tmp_in_x < 0)
        {
         tmp_in_x= 0;
        }
	    const float in_x = tmp_in_x;


        float tmp_in_z = (crop_zdepth > 1)
                            ? z1 * (image_zdepth ) + z * zdepth_scale + zdepth_scale/2 - 0.5
                            : 0.5 * (z1 + z2) * (image_zdepth);
        if (tmp_in_z > image_zdepth - 1)
        {
         tmp_in_z = image_zdepth - 1;
        }
        if (tmp_in_z < 0)
        {
         tmp_in_z= 0;
        }
        const float in_z = tmp_in_z;

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const int front_z_index = floorf(in_z);
        const int back_z_index = ceilf(in_z);
        const float z_lerp = in_z - front_z_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width * image_zdepth;

        // top left front
        atomicAdd(
            pimage + front_z_index + image_zdepth * (left_x_index + image_width * top_y_index),
            (1 - x_lerp) * (1 - z_lerp) * (1 - y_lerp) * grads_ptr[out_idx]   // THIS IS BACKWARD INTERPOL.
        );

        // top left back
        atomicAdd(
            pimage + back_z_index + image_zdepth * (left_x_index + image_width * top_y_index),
            (1 - x_lerp) * (z_lerp) * (1 - y_lerp) * grads_ptr[out_idx]   // THIS IS BACKWARD INTERPOL.
        );

        // top right front
        atomicAdd(
            pimage + front_z_index + image_zdepth * (right_x_index + image_width * top_y_index),
            (x_lerp) * (1 - z_lerp) * (1 - y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

        // top right back
        atomicAdd(
            pimage + back_z_index + image_zdepth * (right_x_index + image_width * top_y_index),
            (x_lerp) * (z_lerp) * (1 - y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

        // bottom left front
        atomicAdd(
            pimage + front_z_index + image_zdepth * (left_x_index + image_width * bottom_y_index),
            (1 - x_lerp) * (1 - z_lerp) * (y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

        // bottom left back
        atomicAdd(
            pimage + back_z_index + image_zdepth * (left_x_index + image_width * bottom_y_index),
            (1 - x_lerp) * (z_lerp) * (y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

        // bottom right front
        atomicAdd(
            pimage + front_z_index + image_zdepth * (right_x_index + image_width * bottom_y_index),
            (x_lerp) * (1 - z_lerp) * (y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

        // bottom right back
        atomicAdd(
            pimage + back_z_index + image_zdepth * (right_x_index + image_width * bottom_y_index),
            (x_lerp) * (z_lerp) * (y_lerp) * grads_ptr[out_idx]   // THIS IS backward INTERPOL.
        );

    }
}



void CropAndResizeLaucher(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_zdepth, int crop_height, int crop_width, int crop_zdepth, int depth,
    float extrapolation_value, float *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_zdepth * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width, image_zdepth,
            crop_height, crop_width, crop_zdepth, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int image_zdepth, int crop_height, int crop_width, int crop_zdepth, int depth,
    float *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * crop_zdepth * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width, image_zdepth,
            crop_height, crop_width, crop_zdepth, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed in Roi Align : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
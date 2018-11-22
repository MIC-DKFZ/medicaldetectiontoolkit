#include <TH/TH.h>
#include <math.h>


int cpu_nms(THLongTensor * keep_out, THLongTensor * num_out, THFloatTensor * boxes, THLongTensor * order, THFloatTensor * areas, float nms_overlap_thresh) {
    // boxes has to be sorted
    THArgCheck(THLongTensor_isContiguous(keep_out), 0, "keep_out must be contiguous");
    THArgCheck(THLongTensor_isContiguous(boxes), 2, "boxes must be contiguous");
    THArgCheck(THLongTensor_isContiguous(order), 3, "order must be contiguous");
    THArgCheck(THLongTensor_isContiguous(areas), 4, "areas must be contiguous");
    // Number of ROIs
    long boxes_num = THFloatTensor_size(boxes, 0);
    long boxes_dim = THFloatTensor_size(boxes, 1);

    long * keep_out_flat = THLongTensor_data(keep_out);
    float * boxes_flat = THFloatTensor_data(boxes);
    long * order_flat = THLongTensor_data(order);
    float * areas_flat = THFloatTensor_data(areas);

    THByteTensor* suppressed = THByteTensor_newWithSize1d(boxes_num);
    THByteTensor_fill(suppressed, 0);
    unsigned char * suppressed_flat =  THByteTensor_data(suppressed);
    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iz1, iz2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2, zz1, zz2;
    float w, h, d;
    float inter, ovr;

    long num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_flat[_i]; // from sorted index to nominal index in boxes list.
        if (suppressed_flat[i] == 1) { //maybe flag for later. overlapping boxes are surpressed.
            continue;
        }
        keep_out_flat[num_to_keep++] = i; //num to keep is read and then increased. the box index i is saved in keep_out.
        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iz1 = boxes_flat[i * boxes_dim + 4];
        iz2 = boxes_flat[i * boxes_dim + 5];
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_flat[_j];
            if (suppressed_flat[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
            zz1 = fmaxf(iz1, boxes_flat[j * boxes_dim + 4]);
            zz2 = fminf(iz2, boxes_flat[j * boxes_dim + 5]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            d = fmaxf(0.0, zz2 - zz1 + 1);
            inter = w * h * d;
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[j] = 1; // can be surpressed because score j < score i (from order: _j = _i + 1 ...)
            }
        }
    }

    long *num_out_flat = THLongTensor_data(num_out);
    *num_out_flat = num_to_keep;
    THByteTensor_free(suppressed);
    return 1;
}
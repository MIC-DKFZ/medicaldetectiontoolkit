#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Unet-like Backbone architecture, with non-parametric heuristics for box detection on semantic segmentation outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.measurements import label as lb
import numpy as np
import utils.exp_utils as utils
import utils.model_utils as mutils


class net(nn.Module):

    def __init__(self, cf, logger):

        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        backbone = utils.import_module('bbone', cf.backbone_path)
        conv = mutils.NDConvGenerator(cf.dim)

        # set operate_stride1=True to generate a unet-like FPN.)
        self.fpn = backbone.FPN(cf, conv, operate_stride1=True).cuda()
        self.conv_final = conv(cf.end_filts, cf.num_seg_classes, ks=1, pad=0, norm=cf.norm, relu=None)

        if self.cf.weight_init is not None:
            logger.info("using pytorch weight init of type {}".format(self.cf.weight_init))
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")


    def forward(self, x):
        """
        forward pass of network.
        :param x: input image. shape (b, c, y, x, (z))
        :return: seg_logits: shape (b, n_classes, y, x, (z))
        :return: out_box_coords: list over n_classes. elements are arrays(b, n_rois, (y1, x1, y2, x2, (z1), (z2)))
        :return: out_max_scores: list over n_classes. elements are arrays(b, n_rois)
        """

        out_features = self.fpn(x)[0]
        seg_logits = self.conv_final(out_features)
        out_box_coords, out_max_scores = [], []
        smax = F.softmax(seg_logits, dim=1).detach().cpu().data.numpy()

        for cl in range(1, len(self.cf.class_dict.keys()) + 1):
            max_scores = [[] for _ in range(x.shape[0])]
            hard_mask = np.copy(smax).argmax(1)
            hard_mask[hard_mask != cl] = 0
            hard_mask[hard_mask == cl] = 1
            # perform connected component analysis on argmaxed predictions,
            # draw boxes around components and return coordinates.
            box_coords, rois = get_coords(hard_mask, self.cf.n_roi_candidates, self.cf.dim)

            # for each object, choose the highest softmax score (in the respective class)
            # of all pixels in the component as object score.
            for bix, broi in enumerate(rois):
                for nix, nroi in enumerate(broi):
                    component_score = np.max(smax[bix, cl][nroi > 0]) if self.cf.aggregation_operation == 'max' \
                        else np.median(smax[bix, cl][nroi > 0])
                    max_scores[bix].append(component_score)
            out_box_coords.append(box_coords)
            out_max_scores.append(max_scores)
        return seg_logits, out_box_coords, out_max_scores


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :param kwargs:
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        seg = batch['seg']
        var_img = torch.FloatTensor(img).cuda()
        var_seg = torch.FloatTensor(seg).cuda().long()
        var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(seg, self.cf.num_seg_classes)).cuda()
        results_dict = {}
        seg_logits, box_coords, max_scores = self.forward(var_img)

        results_dict['boxes'] = [[] for _ in range(img.shape[0])]
        for cix in range(len(self.cf.class_dict.keys())):
            for bix in range(img.shape[0]):
                for rix in range(len(max_scores[cix][bix])):
                    if max_scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                                           'box_score': max_scores[cix][bix][rix],
                                                           'box_pred_class_id': cix + 1,  # add 0 for background.
                                                           'box_type': 'det'})
        if "roi_labels" in batch.keys():
            raise Exception("Key for roi-wise class targets changed in v0.1.0 from 'roi_labels' to 'class_target'.\n"
                            "If you use DKFZ's batchgenerators, please make sure you run version >= 0.20.1.")
        for bix in range(img.shape[0]):
            for tix in range(len(batch['bb_target'][bix])):
                results_dict['boxes'][bix].append({'box_coords': batch['bb_target'][bix][tix],
                                                   'box_label': batch['class_target'][bix][tix],
                                                   'box_type': 'gt'})

        # compute segmentation loss as either weighted cross entropy, dice loss, or the sum of both.
        loss = torch.FloatTensor([0]).cuda()
        if self.cf.seg_loss_mode == 'dice' or self.cf.seg_loss_mode == 'dice_wce':
            loss += 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1), var_seg_ohe,
                                          false_positive_weight=float(self.cf.fp_dice_weight))

        if self.cf.seg_loss_mode == 'wce' or self.cf.seg_loss_mode == 'dice_wce':
            loss += F.cross_entropy(seg_logits, var_seg[:, 0], weight=torch.tensor(self.cf.wce_weights).float().cuda())

        results_dict['seg_preds'] = np.argmax(F.softmax(seg_logits, 1).cpu().data.numpy(), 1)[:, np.newaxis]
        results_dict['torch_loss'] = loss
        results_dict['monitor_values'] = {'loss': loss.item()}
        results_dict['logger_string'] = "loss: {0:.2f}".format(loss.item())


        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :param kwargs:
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
        """
        img = batch['data']
        var_img = torch.FloatTensor(img).cuda()
        seg_logits, box_coords, max_scores = self.forward(var_img)

        results_dict = {}
        results_dict['boxes'] = [[] for _ in range(img.shape[0])]
        for cix in range(len(self.cf.class_dict.keys())):
            for bix in range(img.shape[0]):
                for rix in range(len(max_scores[cix][bix])):
                    if max_scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                                           'box_score': max_scores[cix][bix][rix],
                                                           'box_pred_class_id': cix + 1,  # add 0 for background.
                                                           'box_type': 'det'})

        results_dict['seg_preds'] = np.argmax(F.softmax(seg_logits, 1).cpu().data.numpy(), 1)[:, np.newaxis].astype('uint8')
        return results_dict



def get_coords(binary_mask, n_components, dim):
    """
    loops over batch to perform connected component analysis on binary input mask. computes box coordiantes around
    n_components - biggest components (rois).
    :param binary_mask: (b, y, x, (z)). binary mask for one specific foreground class.
    :param n_components: int. number of components to extract per batch element and class.
    :return: coords (b, n, (y1, x1, y2, x2, (z1), (z2))
    :return: batch_components (b, n, (y1, x1, y2, x2, (z1), (z2))
    """
    binary_mask = binary_mask.astype('uint8')
    batch_coords = []
    batch_components = []
    for ix, b in enumerate(binary_mask):
        clusters, n_cands = lb(b)  # peforms connected component analysis.
        uniques, counts = np.unique(clusters, return_counts=True)
        # only keep n_components largest components.
        keep_uniques = uniques[1:][np.argsort(counts[1:])[::-1]][:n_components]
        # separate clusters and concat.
        p_components = np.array([(clusters == ii) * 1 for ii in keep_uniques])
        p_coords = []
        if p_components.shape[0] > 0:
            for roi in p_components:
                mask_ixs = np.argwhere(roi != 0)

                # get coordinates around component.
                roi_coords = [np.min(mask_ixs[:, 0]) - 1, np.min(mask_ixs[:, 1]) - 1, np.max(mask_ixs[:, 0]) + 1,
                              np.max(mask_ixs[:, 1]) + 1]
                if dim == 3:
                    roi_coords += [np.min(mask_ixs[:, 2]), np.max(mask_ixs[:, 2])+1]
                p_coords.append(roi_coords)

            p_coords = np.array(p_coords)

            # clip coords.
            p_coords[p_coords < 0] = 0
            p_coords[:, :4][p_coords[:, :4] > binary_mask.shape[-2]] = binary_mask.shape[-2]
            if dim == 3:
                p_coords[:, 4:][p_coords[:, 4:] > binary_mask.shape[-1]] = binary_mask.shape[-1]

        batch_coords.append(p_coords)
        batch_components.append(p_components)
    return batch_coords, batch_components


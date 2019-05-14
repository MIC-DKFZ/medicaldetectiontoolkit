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

import os
import numpy as np
import torch
from scipy.stats import norm
from collections import OrderedDict
from multiprocessing import Pool
import pickle
import pandas as pd


class Predictor:
    """
    Prediction pipeline:
    - receives a patched patient image (n_patches, c, y, x, (z)) from patient data loader.
    - forwards patches through model in chunks of batch_size. (method: batch_tiling_forward)
    - unmolds predictions (boxes and segmentations) to original patient coordinates. (method: spatial_tiling_forward)

    Ensembling (mode == 'test'):
    - for inference, forwards 4 mirrored versions of image to through model and unmolds predictions afterwards
      accordingly (method: data_aug_forward)
    - for inference, loads multiple parameter-sets of the trained model corresponding to different epochs. for each
      parameter-set loops over entire test set, runs prediction pipeline for each patient. (method: predict_test_set)

    Consolidation of predictions:
    - consolidates a patient's predictions (boxes, segmentations) collected over patches, data_aug- and temporal ensembling,
      performs clustering and weighted averaging (external function: apply_wbc_to_patient) to obtain consistent outptus.
    - for 2D networks, consolidates box predictions to 3D cubes via clustering (adaption of non-maximum surpression).
      (external function: merge_2D_to_3D_preds_per_patient)

    Ground truth handling:
    - dissmisses any ground truth boxes returned by the model (happens in validation mode, patch-based groundtruth)
    - if provided by data loader, adds 3D ground truth to the final predictions to be passed to the evaluator.
    """
    def __init__(self, cf, net, logger, mode):

        self.cf = cf
        self.logger = logger

        # mode is 'val' for patient-based validation/monitoring and 'test' for inference.
        self.mode = mode

        # model instance. In validation mode, contains parameters of current epoch.
        self.net = net

        # rank of current epoch loaded (for temporal averaging). this info is added to each prediction,
        # for correct weighting during consolidation.
        self.rank_ix = '0'

        # number of ensembled models. used to calculate the number of expected predictions per position
        # during consolidation of predictions. Default is 1 (no ensembling, e.g. in validation).
        self.n_ens = 1

        if self.mode == 'test':
            try:
                self.epoch_ranking = np.load(os.path.join(self.cf.fold_dir, 'epoch_ranking.npy'))[:cf.test_n_epochs]
            except:
                raise RuntimeError('no epoch ranking file in fold directory. '
                                   'seems like you are trying to run testing without prior training...')
            self.n_ens = cf.test_n_epochs
            if self.cf.test_aug:
                self.n_ens *= 4


    def predict_patient(self, batch):
        """
        predicts one patient.
        called either directly via loop over validation set in exec.py (mode=='val')
        or from self.predict_test_set (mode=='test).
        in val mode:  adds 3D ground truth info to predictions and runs consolidation and 2Dto3D merging of predictions.
        in test mode: returns raw predictions (ground truth addition, consolidation, 2D to 3D merging are
                      done in self.predict_test_set, because patient predictions across several epochs might be needed
                      to be collected first, in case of temporal ensembling).
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                            (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - monitor_values (only in validation mode)
        """
        self.logger.info('evaluating patient {} for fold {} '.format(batch['pid'], self.cf.fold))

        # True if patient is provided in patches and predictions need to be tiled.
        self.patched_patient = True if 'patch_crop_coords' in list(batch.keys()) else False

        # forward batch through prediction pipeline.
        results_dict = self.data_aug_forward(batch)

        if self.mode == 'val':
            for b in range(batch['patient_bb_target'].shape[0]):
                for t in range(len(batch['patient_bb_target'][b])):
                    results_dict['boxes'][b].append({'box_coords': batch['patient_bb_target'][b][t],
                                                     'box_label': batch['patient_roi_labels'][b][t],
                                                     'box_type': 'gt'})

            if self.patched_patient:
                wcs_input = [results_dict['boxes'], 'dummy_pid', self.cf.class_dict, self.cf.wcs_iou, self.n_ens]
                results_dict['boxes'] = apply_wbc_to_patient(wcs_input)[0]

            if self.cf.merge_2D_to_3D_preds:
                merge_dims_inputs = [results_dict['boxes'], 'dummy_pid', self.cf.class_dict, self.cf.merge_3D_iou]
                results_dict['boxes'] = merge_2D_to_3D_preds_per_patient(merge_dims_inputs)[0]

        return results_dict


    def predict_test_set(self, batch_gen, return_results=True):
        """
        wrapper around test method, which loads multiple (or one) epoch parameters (temporal ensembling), loops through
        the test set and collects predictions per patient. Also flattens the results per patient and epoch
        and adds optional ground truth boxes for evaluation. Saves out the raw result list for later analysis and
        optionally consolidates and returns predictions immediately.
        :return: (optionally) list_of_results_per_patient: list over patient results. each entry is a dict with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                            (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': not implemented yet. todo for evaluation of instance/semantic segmentation.
        """
        dict_of_patient_results = OrderedDict()

        # get paths of all parameter sets to be loaded for temporal ensembling. (or just one for no temp. ensembling).
        weight_paths = [os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(epoch), 'params.pth') for epoch in
                        self.epoch_ranking]

        for rank_ix, weight_path in enumerate(weight_paths):

            self.logger.info(('tmp ensembling over rank_ix:{} epoch:{}'.format(rank_ix, weight_path)))
            self.net.load_state_dict(torch.load(weight_path))
            self.net.eval()
            self.rank_ix = str(rank_ix)  # get string of current rank for unique patch ids.

            with torch.no_grad():
                for _ in range(batch_gen['n_test']):

                    batch = next(batch_gen['test'])

                    # store batch info in patient entry of results dict.
                    if rank_ix == 0:
                        dict_of_patient_results[batch['pid']] = {}
                        dict_of_patient_results[batch['pid']]['results_list'] = []
                        dict_of_patient_results[batch['pid']]['patient_bb_target'] = batch['patient_bb_target']
                        dict_of_patient_results[batch['pid']]['patient_roi_labels'] = batch['patient_roi_labels']

                    # call prediction pipeline and store results in dict.
                    results_dict = self.predict_patient(batch)
                    dict_of_patient_results[batch['pid']]['results_list'].append(results_dict['boxes'])


        self.logger.info('finished predicting test set. starting post-processing of predictions.')
        list_of_results_per_patient = []

        # loop over patients again to flatten results across epoch predictions.
        # if provided, add ground truth boxes for evaluation.
        for pid, p_dict in dict_of_patient_results.items():

            tmp_ens_list = p_dict['results_list']
            results_dict = {}
            # collect all boxes/seg_preds of same batch_instance over temporal instances.
            results_dict['boxes'] = [[item for d in tmp_ens_list for item in d[batch_instance]]
                                     for batch_instance in range(len(tmp_ens_list[0]))]

            # TODO return for instance segmentation:
            # results_dict['seg_preds'] = np.mean(results_dict['seg_preds'], 1)[:, None]
            # results_dict['seg_preds'] = np.array([[item for d in tmp_ens_list for item in d['seg_preds'][batch_instance]]
            #                                       for batch_instance in range(len(tmp_ens_list[0]['boxes']))])

            # add 3D ground truth boxes for evaluation.
            for b in range(p_dict['patient_bb_target'].shape[0]):
                for t in range(len(p_dict['patient_bb_target'][b])):
                    results_dict['boxes'][b].append({'box_coords': p_dict['patient_bb_target'][b][t],
                                                     'box_label': p_dict['patient_roi_labels'][b][t],
                                                     'box_type': 'gt'})

            list_of_results_per_patient.append([results_dict['boxes'], pid])

        # save out raw predictions.
        out_string = 'raw_pred_boxes_hold_out_list' if self.cf.hold_out_test_set else 'raw_pred_boxes_list'
        with open(os.path.join(self.cf.fold_dir, '{}.pickle'.format(out_string)), 'wb') as handle:
            pickle.dump(list_of_results_per_patient, handle)

        if return_results:

            # consolidate predictions.
            self.logger.info('applying wcs to test set predictions with iou = {} and n_ens = {}.'.format(
                self.cf.wcs_iou, self.n_ens))
            pool = Pool(processes=6)
            mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.wcs_iou, self.n_ens] for ii in list_of_results_per_patient]
            list_of_results_per_patient = pool.map(apply_wbc_to_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()

            # merge 2D boxes to 3D cubes. (if model predicts 2D but evaluation is run in 3D)
            if self.cf.merge_2D_to_3D_preds:
                self.logger.info('applying 2Dto3D merging to test set predictions with iou = {}.'.format(self.cf.merge_3D_iou))
                pool = Pool(processes=6)
                mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.merge_3D_iou] for ii in list_of_results_per_patient]
                list_of_results_per_patient = pool.map(merge_2D_to_3D_preds_per_patient, mp_inputs, chunksize=1)
                pool.close()
                pool.join()

            return list_of_results_per_patient


    def load_saved_predictions(self, apply_wbc=False):
        """
        loads raw predictions saved by self.predict_test_set. consolidates and merges 2D boxes to 3D cubes for evaluation.
        (if model predicts 2D but evaluation is run in 3D)
        :return: (optionally) list_of_results_per_patient: list over patient results. each entry is a dict with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                            (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': not implemented yet. todo for evaluation of instance/semantic segmentation.
        """

        # load predictions for a single test-set fold.
        if not self.cf.hold_out_test_set:
            with open(os.path.join(self.cf.fold_dir, 'raw_pred_boxes_list.pickle'), 'rb') as handle:
                list_of_results_per_patient = pickle.load(handle)
            da_factor = 4 if self.cf.test_aug else 1
            n_ens = self.cf.test_n_epochs * da_factor
            self.logger.info('loaded raw test set predictions with n_patients = {} and n_ens = {}'.format(
                len(list_of_results_per_patient), n_ens))

        # if hold out test set was perdicted, aggregate predictions of all trained models
        # corresponding to all CV-folds and flatten them.
        else:
            boxes_list = []
            for fold in self.cf.folds:
                fold_dir = os.path.join(self.cf.exp_dir, 'fold_{}'.format(fold))
                with open(os.path.join(fold_dir, 'raw_pred_boxes_hold_out_list.pickle'), 'rb') as handle:
                    fold_list = pickle.load(handle)
                    pids = [ii[1] for ii in fold_list]
                    boxes_list.append([ii[0] for ii in fold_list])
            list_of_results_per_patient = [[[[box for fold_list in boxes_list for box in fold_list[pix][0]
                                              if box['box_type'] == 'det']], pid] for pix, pid in enumerate(pids)]
            da_factor = 4 if self.cf.test_aug else 1
            n_ens = self.cf.test_n_epochs * da_factor * len(self.cf.folds)

        # consolidate predictions.
        if apply_wbc:
            self.logger.info('applying wcs to test set predictions with iou = {} and n_ens = {}.'.format(
                self.cf.wcs_iou, n_ens))
            pool = Pool(processes=6)
            mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.wcs_iou, n_ens] for ii in list_of_results_per_patient]
            list_of_results_per_patient = pool.map(apply_wbc_to_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()
        else:
            list_of_results_per_patient = list_of_results_per_patient

        # merge 2D box predictions to 3D cubes (if model predicts 2D but evaluation is run in 3D)
        if self.cf.merge_2D_to_3D_preds:
            self.logger.info(
                'applying 2Dto3D merging to test set predictions with iou = {}.'.format(self.cf.merge_3D_iou))
            pool = Pool(processes=6)
            mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.merge_3D_iou] for ii in list_of_results_per_patient]
            list_of_results_per_patient = pool.map(merge_2D_to_3D_preds_per_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()

        return list_of_results_per_patient


    def data_aug_forward(self, batch):
        """
        in val_mode: passes batch through to spatial_tiling method without data_aug.
        in test_mode: if cf.test_aug is set in configs, createst 4 mirrored versions of the input image,
        passes all of them to the next processing step (spatial_tiling method) and re-transforms returned predictions
        to original image version.
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - monitor_values (only in validation mode)
        """
        patch_crops = batch['patch_crop_coords'] if self.patched_patient else None
        results_list = [self.spatial_tiling_forward(batch, patch_crops)]
        org_img_shape = batch['original_img_shape']

        if self.mode == 'test' and self.cf.test_aug:

            if self.patched_patient:
                # apply mirror transformations to patch-crop coordinates, for correct tiling in spatial_tiling method.
                mirrored_patch_crops = get_mirrored_patch_crops(patch_crops, batch['original_img_shape'])
            else:
                mirrored_patch_crops = [None] * 3

            img = np.copy(batch['data'])

            # first mirroring: y-axis.
            batch['data'] = np.flip(img, axis=2).copy()
            chunk_dict = self.spatial_tiling_forward(batch, mirrored_patch_crops[0], n_aug='1')
            # re-transform coordinates.
            for ix in range(len(chunk_dict['boxes'])):
                for boxix in range(len(chunk_dict['boxes'][ix])):
                    coords = chunk_dict['boxes'][ix][boxix]['box_coords'].copy()
                    coords[0] = org_img_shape[2] - chunk_dict['boxes'][ix][boxix]['box_coords'][2]
                    coords[2] = org_img_shape[2] - chunk_dict['boxes'][ix][boxix]['box_coords'][0]
                    assert coords[2] >= coords[0], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    assert coords[3] >= coords[1], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    chunk_dict['boxes'][ix][boxix]['box_coords'] = coords
            # re-transform segmentation predictions.
            chunk_dict['seg_preds'] = np.flip(chunk_dict['seg_preds'], axis=2)
            results_list.append(chunk_dict)

            # second mirroring: x-axis.
            batch['data'] = np.flip(img, axis=3).copy()
            chunk_dict = self.spatial_tiling_forward(batch, mirrored_patch_crops[1], n_aug='2')
            # re-transform coordinates.
            for ix in range(len(chunk_dict['boxes'])):
                for boxix in range(len(chunk_dict['boxes'][ix])):
                    coords = chunk_dict['boxes'][ix][boxix]['box_coords'].copy()
                    coords[1] = org_img_shape[3] - chunk_dict['boxes'][ix][boxix]['box_coords'][3]
                    coords[3] = org_img_shape[3] - chunk_dict['boxes'][ix][boxix]['box_coords'][1]
                    assert coords[2] >= coords[0], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    assert coords[3] >= coords[1], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    chunk_dict['boxes'][ix][boxix]['box_coords'] = coords
            # re-transform segmentation predictions.
            chunk_dict['seg_preds'] = np.flip(chunk_dict['seg_preds'], axis=3)
            results_list.append(chunk_dict)

            # third mirroring: y-axis and x-axis.
            batch['data'] = np.flip(np.flip(img, axis=2), axis=3).copy()
            chunk_dict = self.spatial_tiling_forward(batch, mirrored_patch_crops[2], n_aug='3')
            # re-transform coordinates.
            for ix in range(len(chunk_dict['boxes'])):
                for boxix in range(len(chunk_dict['boxes'][ix])):
                    coords = chunk_dict['boxes'][ix][boxix]['box_coords'].copy()
                    coords[0] = org_img_shape[2] - chunk_dict['boxes'][ix][boxix]['box_coords'][2]
                    coords[2] = org_img_shape[2] - chunk_dict['boxes'][ix][boxix]['box_coords'][0]
                    coords[1] = org_img_shape[3] - chunk_dict['boxes'][ix][boxix]['box_coords'][3]
                    coords[3] = org_img_shape[3] - chunk_dict['boxes'][ix][boxix]['box_coords'][1]
                    assert coords[2] >= coords[0], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    assert coords[3] >= coords[1], [coords, chunk_dict['boxes'][ix][boxix]['box_coords'].copy()]
                    chunk_dict['boxes'][ix][boxix]['box_coords'] = coords
            # re-transform segmentation predictions.
            chunk_dict['seg_preds'] = np.flip(np.flip(chunk_dict['seg_preds'], axis=2), axis=3).copy()
            results_list.append(chunk_dict)

            batch['data'] = img

        # aggregate all boxes/seg_preds per batch element from data_aug predictions.
        results_dict = {}
        results_dict['boxes'] = [[item for d in results_list for item in d['boxes'][batch_instance]]
                                 for batch_instance in range(org_img_shape[0])]
        results_dict['seg_preds'] = np.array([[item for d in results_list for item in d['seg_preds'][batch_instance]]
                                              for batch_instance in range(org_img_shape[0])])
        if self.mode == 'val':
            results_dict['monitor_values'] = results_list[0]['monitor_values']

        return results_dict


    def spatial_tiling_forward(self, batch, patch_crops=None, n_aug='0'):
        """
        forwards batch to batch_tiling_forward method and receives and returns a dictionary with results.
        if patch-based prediction, the results received from batch_tiling_forward will be on a per-patch-basis.
        this method uses the provided patch_crops to re-transform all predictions to whole-image coordinates.
        Patch-origin information of all box-predictions will be needed for consolidation, hence it is stored as
        'patch_id', which is a unique string for each patch (also takes current data aug and temporal epoch instances
        into account). all box predictions get additional information about the amount overlapping patches at the
        respective position (used for consolidation).
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - monitor_values (only in validation mode)
        """
        if patch_crops is not None:

            patches_dict = self.batch_tiling_forward(batch)

            results_dict = {'boxes': [[] for _ in range(batch['original_img_shape'][0])]}

            # instanciate segemntation output array. Will contain averages over patch predictions.
            out_seg_preds = np.zeros(batch['original_img_shape'], dtype=np.float16)[:, 0][:, None]
            # counts patch instances per pixel-position.
            patch_overlap_map = np.zeros_like(out_seg_preds, dtype='uint8')

            #unmold segmentation outputs. loop over patches.
            for pix, pc in enumerate(patch_crops):
                if self.cf.dim == 3:
                    out_seg_preds[:, :, pc[0]:pc[1], pc[2]:pc[3], pc[4]:pc[5]] += patches_dict['seg_preds'][pix][None]
                    patch_overlap_map[:, :, pc[0]:pc[1], pc[2]:pc[3], pc[4]:pc[5]] += 1
                else:
                    out_seg_preds[pc[4]:pc[5], :, pc[0]:pc[1], pc[2]:pc[3], ] += patches_dict['seg_preds'][pix]
                    patch_overlap_map[pc[4]:pc[5], :, pc[0]:pc[1], pc[2]:pc[3], ] += 1

            # take mean in overlapping areas.
            out_seg_preds[patch_overlap_map > 0] /= patch_overlap_map[patch_overlap_map > 0]
            results_dict['seg_preds'] = out_seg_preds

            # unmold box outputs. loop over patches.
            for pix, pc in enumerate(patch_crops):
                patch_boxes = patches_dict['boxes'][pix]

                for box in patch_boxes:

                    # add unique patch id for consolidation of predictions.
                    box['patch_id'] = self.rank_ix + '_' + n_aug + '_' + str(pix)

                    # boxes from the edges of a patch have a lower prediction quality, than the ones at patch-centers.
                    # hence they will be downweighted for consolidation, using the 'box_patch_center_factor', which is
                    # obtained by a normal distribution over positions in the patch and average over spatial dimensions.
                    # Also the info 'box_n_overlaps' is stored for consolidation, which depicts the amount over
                    # overlapping patches at the box's position.
                    c = box['box_coords']
                    box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                    if self.cf.dim == 3:
                        box_centers.append((c[4] + c[5]) / 2)
                    box['box_patch_center_factor'] = np.mean(
                        [norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in
                         zip(box_centers, np.array(self.cf.patch_size) / 2)])
                    if self.cf.dim == 3:
                        c += np.array([pc[0], pc[2], pc[0], pc[2], pc[4], pc[4]])
                        int_c = [int(np.floor(ii)) if ix%2 == 0 else int(np.ceil(ii)) for ix, ii in enumerate(c)]
                        box['box_n_overlaps'] = np.mean(patch_overlap_map[:, :, int_c[1]:int_c[3], int_c[0]:int_c[2], int_c[4]:int_c[5]])
                        results_dict['boxes'][0].append(box)
                    else:
                        c += np.array([pc[0], pc[2], pc[0], pc[2]])
                        int_c = [int(np.floor(ii)) if ix % 2 == 0 else int(np.ceil(ii)) for ix, ii in enumerate(c)]
                        box['box_n_overlaps'] = np.mean(patch_overlap_map[pc[4], :, int_c[1]:int_c[3], int_c[0]:int_c[2]])
                        results_dict['boxes'][pc[4]].append(box)

            if self.mode == 'val':
                results_dict['monitor_values'] = patches_dict['monitor_values']

        # if predictions are not patch-based:
        # add patch-origin info to boxes (entire image is the same patch with overlap=1) and return results.
        else:
            results_dict = self.batch_tiling_forward(batch)
            for b in results_dict['boxes']:
                for box in b:
                    box['box_patch_center_factor'] = 1
                    box['box_n_overlaps'] = 1
                    box['patch_id'] = self.rank_ix + '_' + n_aug

        return results_dict


    def batch_tiling_forward(self, batch):
        """
        calls the actual network forward method. in patch-based prediction, the batch dimension might be overladed
        with n_patches >> batch_size, which would exceed gpu memory. In this case, batches are processed in chunks of
        batch_size. validation mode calls the train method to monitor losses (returned ground truth objects are discarded).
        test mode calls the test forward method, no ground truth required / involved.
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - monitor_values (only in validation mode)
        """
        self.logger.info('forwarding (patched) patient with shape: {}'.format(batch['data'].shape))

        img = batch['data']

        if img.shape[0] <= self.cf.batch_size:

            if self.mode == 'val':
                # call training method to monitor losses
                results_dict = self.net.train_forward(batch, is_validation=True)
                # discard returned ground-truth boxes (also training info boxes).
                results_dict['boxes'] = [[box for box in b if box['box_type'] == 'det'] for b in results_dict['boxes']]
            else:
                results_dict = self.net.test_forward(batch, return_masks=self.cf.return_masks_in_test)

        else:
            split_ixs = np.split(np.arange(img.shape[0]), np.arange(img.shape[0])[::self.cf.batch_size])
            chunk_dicts = []
            for chunk_ixs in split_ixs[1:]:  # first split is elements before 0, so empty
                b = {k: batch[k][chunk_ixs] for k in batch.keys()
                     if (isinstance(batch[k], np.ndarray) and batch[k].shape[0] == img.shape[0])}
                if self.mode == 'val':
                    chunk_dicts += [self.net.train_forward(b, is_validation=True)]
                else:
                    chunk_dicts += [self.net.test_forward(b, return_masks=self.cf.return_masks_in_test)]


            results_dict = {}
            # flatten out batch elements from chunks ([chunk, chunk] -> [b, b, b, b, ...])
            results_dict['boxes'] = [item for d in chunk_dicts for item in d['boxes']]
            results_dict['seg_preds'] = np.array([item for d in chunk_dicts for item in d['seg_preds']])

            if self.mode == 'val':
                # estimate metrics by mean over batch_chunks. Most similar to training metrics.
                results_dict['monitor_values'] = \
                    {k:np.mean([d['monitor_values'][k] for d in chunk_dicts])
                     for k in chunk_dicts[0]['monitor_values'].keys()}
                # discard returned ground-truth boxes (also training info boxes).
                results_dict['boxes'] = [[box for box in b if box['box_type'] == 'det'] for b in results_dict['boxes']]

        return results_dict



def apply_wbc_to_patient(inputs):
    """
    wrapper around prediction box consolidation: weighted cluster scoring (wcs). processes a single patient.
    loops over batch elements in patient results (1 in 3D, slices in 2D) and foreground classes,
    aggregates and stores results in new list.
    :return. patient_results_list: list over batch elements. each element is a list over boxes, where each box is
                                 one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D
                                 predictions, and a dummy batch dimension of 1 for 3D predictions.
    :return. pid: string. patient id.
    """
    in_patient_results_list, pid, class_dict, wcs_iou, n_ens = inputs
    out_patient_results_list = [[] for _ in range(len(in_patient_results_list))]

    for bix, b in enumerate(in_patient_results_list):

        for cl in list(class_dict.keys()):

            boxes = [(ix, box) for ix, box in enumerate(b) if (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            box_coords = np.array([b[1]['box_coords'] for b in boxes])
            box_scores = np.array([b[1]['box_score'] for b in boxes])
            box_center_factor = np.array([b[1]['box_patch_center_factor'] for b in boxes])
            box_n_overlaps = np.array([b[1]['box_n_overlaps'] for b in boxes])
            box_patch_id = np.array([b[1]['patch_id'] for b in boxes])

            if 0 not in box_scores.shape:
                keep_scores, keep_coords = weighted_box_clustering(
                    np.concatenate((box_coords, box_scores[:, None], box_center_factor[:, None],
                                    box_n_overlaps[:, None]), axis=1), box_patch_id, wcs_iou, n_ens)

                for boxix in range(len(keep_scores)):
                    out_patient_results_list[bix].append({'box_type': 'det', 'box_coords': keep_coords[boxix],
                                             'box_score': keep_scores[boxix], 'box_pred_class_id': cl})

        # add gt boxes back to new output list.
        out_patient_results_list[bix].extend([box for box in b if box['box_type'] == 'gt'])

    return [out_patient_results_list, pid]



def merge_2D_to_3D_preds_per_patient(inputs):
    """
    wrapper around 2Dto3D merging operation. Processes a single patient. Takes 2D patient results (slices in batch dimension)
    and returns 3D patient results (dummy batch dimension of 1). Applies an adaption of Non-Maximum Surpression
    (Detailed methodology is described in nms_2to3D).
    :return. results_dict_boxes: list over batch elements (1 in 3D). each element is a list over boxes, where each box is
                                 one dictionary: [[box_0, ...], [box_n,...]].
    :return. pid: string. patient id.
    """
    in_patient_results_list, pid, class_dict, merge_3D_iou = inputs
    out_patient_results_list = []

    for cl in list(class_dict.keys()):
        boxes, slice_ids = [], []
        # collect box predictions over batch dimension (slices) and store slice info as slice_ids.
        for bix, b in enumerate(in_patient_results_list):
            det_boxes = [(ix, box) for ix, box in enumerate(b) if
                     (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            boxes += det_boxes
            slice_ids += [bix] * len(det_boxes)

        box_coords = np.array([b[1]['box_coords'] for b in boxes])
        box_scores = np.array([b[1]['box_score'] for b in boxes])
        slice_ids = np.array(slice_ids)

        if 0 not in box_scores.shape:
            keep_ix, keep_z = nms_2to3D(
                np.concatenate((box_coords, box_scores[:, None], slice_ids[:, None]), axis=1), merge_3D_iou)
        else:
            keep_ix, keep_z = [], []

        # store kept predictions in new results list and add corresponding z-dimension info to coordinates.
        for kix, kz in zip(keep_ix, keep_z):
            out_patient_results_list.append({'box_type': 'det', 'box_coords': list(box_coords[kix]) + kz,
                                             'box_score': box_scores[kix], 'box_pred_class_id': cl})

    out_patient_results_list += [box for b in in_patient_results_list for box in b if box['box_type'] == 'gt']
    out_patient_results_list = [out_patient_results_list] # add dummy batch dimension 1 for 3D.

    return [out_patient_results_list, pid]



def weighted_box_clustering(dets, box_patch_id, thresh, n_ens):
    """
    consolidates overlapping predictions resulting from patch overlaps, test data augmentations and temporal ensembling.
    clusters predictions together with iou > thresh (like in NMS). Output score and coordinate for one cluster are the
    average weighted by individual patch center factors (how trustworthy is this candidate measured by how centered
    its position the patch is) and the size of the corresponding box.
    The number of expected predictions at a position is n_data_aug * n_temp_ens * n_overlaps_at_position
    (1 prediction per unique patch). Missing predictions at a cluster position are defined as the number of unique
    patches in the cluster, which did not contribute any predict any boxes.
    :param dets: (n_dets, (y1, x1, y2, x2, (z1), (z2), scores, box_pc_facts, box_n_ovs)
    :param thresh: threshold for iou_matching.
    :param n_ens: number of models, that are ensembled. (-> number of expected predicitions per position)
    :return: keep_scores: (n_keep)  new scores of boxes to be kept.
    :return: keep_coords: (n_keep, (y1, x1, y2, x2, (z1), (z2)) new coordinates of boxes to be kept.
    """
    dim = 2 if dets.shape[1] == 7 else 3
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, -3]
    box_pc_facts = dets[:, -2]
    box_n_ovs = dets[:, -1]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    if dim == 3:
        z1 = dets[:, 4]
        z2 = dets[:, 5]
        areas *= (z2 - z1 + 1)

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]

    keep = []
    keep_scores = []
    keep_coords = []

    while order.size > 0:
        i = order[0]  # higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if dim == 3:
            zz1 = np.maximum(z1[i], z1[order])
            zz2 = np.minimum(z2[i], z2[order])
            d = np.maximum(0.0, zz2 - zz1 + 1)
            inter *= d

        # overall between currently highest scoring box and all boxes.
        ovr = inter / (areas[i] + areas[order] - inter)

        # get all the predictions that match the current box to build one cluster.
        matches = np.argwhere(ovr > thresh)

        match_n_ovs = box_n_ovs[order[matches]]
        match_pc_facts = box_pc_facts[order[matches]]
        match_patch_id = box_patch_id[order[matches]]
        match_ov_facts = ovr[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]

        # weight all socres in cluster by patch factors, and size.
        match_score_weights = match_ov_facts * match_areas * match_pc_facts
        match_scores *= match_score_weights

        # for the weigted average, scores have to be divided by the number of total expected preds at the position
        # of the current cluster. 1 Prediction per patch is expected. therefore, the number of ensembled models is
        # multiplied by the mean overlaps of  patches at this position (boxes of the cluster might partly be
        # in areas of different overlaps).
        n_expected_preds = n_ens * np.mean(match_n_ovs)

        # the number of missing predictions is obtained as the number of patches,
        # which did not contribute any prediction to the current cluster.
        n_missing_preds = np.max((0, n_expected_preds - np.unique(match_patch_id).shape[0]))

        # missing preds are given the mean weighting
        # (expected prediction is the mean over all predictions in cluster).
        denom = np.sum(match_score_weights) + n_missing_preds * np.mean(match_score_weights)

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / denom

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores)]
        if dim == 3:
            avg_coords.append(np.sum(z1[order[matches]] * match_scores) / np.sum(match_scores))
            avg_coords.append(np.sum(z2[order[matches]] * match_scores) / np.sum(match_scores))

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.01:
            keep_scores.append(avg_score)
            keep_coords.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(ovr <= thresh)[0]
        order = order[inds]

    return keep_scores, keep_coords



def nms_2to3D(dets, thresh):
    """
    Merges 2D boxes to 3D cubes. Therefore, boxes of all slices are projected into one slices. An adaptation of Non-maximum surpression
    is applied, where clusters are found (like in NMS) with an extra constrained, that surpressed boxes have to have 'connected'
    z-coordinates w.r.t the core slice (cluster center, highest scoring box). 'connected' z-coordinates are determined
    as the z-coordinates with predictions until the first coordinate, where no prediction was found.

    example: a cluster of predictions was found overlap > iou thresh in xy (like NMS). The z-coordinate of the highest
    scoring box is 50. Other predictions have 23, 46, 48, 49, 51, 52, 53, 56, 57.
    Only the coordinates connected with 50 are clustered to one cube: 48, 49, 51, 52, 53. (46 not because nothing was
    found in 47, so 47 is a 'hole', which interrupts the connection). Only the boxes corresponding to these coordinates
    are surpressed. All others are kept for building of further clusters.

    This algorithm works better with a certain min_confidence of predictions, because low confidence (e.g. noisy/cluttery)
    predictions can break the relatively strong assumption of defining cubes' z-boundaries at the first 'hole' in the cluster.

    :param dets: (n_detections, (y1, x1, y2, x2, scores, slice_id)
    :param thresh: iou matchin threshold (like in NMS).
    :return: keep: (n_keep) 1D tensor of indices to be kept.
    :return: keep_z: (n_keep, [z1, z2]) z-coordinates to be added to boxes, which are kept in order to form cubes.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, -2]
    slice_id = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    keep_z = []

    while order.size > 0:  # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
        i = order[0]  # pop higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order] - inter)
        matches = np.argwhere(ovr > thresh)  # get all the elements that match the current box and have a lower score

        slice_ids = slice_id[order[matches]]
        core_slice = slice_id[int(i)]
        upper_wholes = [ii for ii in np.arange(core_slice, np.max(slice_ids)) if ii not in slice_ids]
        lower_wholes = [ii for ii in np.arange(np.min(slice_ids), core_slice) if ii not in slice_ids]
        max_valid_slice_id = np.min(upper_wholes) if len(upper_wholes) > 0 else np.max(slice_ids)
        min_valid_slice_id = np.max(lower_wholes) if len(lower_wholes) > 0 else np.min(slice_ids)
        z_matches = matches[(slice_ids <= max_valid_slice_id) & (slice_ids >= min_valid_slice_id)]

        z1 = np.min(slice_id[order[z_matches]]) - 1
        z2 = np.max(slice_id[order[z_matches]]) + 1

        keep.append(i)
        keep_z.append([z1, z2])
        order = np.delete(order, z_matches, axis=0)

    return keep, keep_z



def get_mirrored_patch_crops(patch_crops, org_img_shape):
    """
    apply 3 mirrror transformations (x-axis, y-axis, x&y-axis)
    to given patch crop coordinates and return the transformed coordinates.
    Handles 2D and 3D coordinates.
    :param patch_crops: list of crops: each element is a list of coordinates for given crop [[y1, x1, ...], [y1, x1, ..]]
    :param org_img_shape: shape of patient volume used as world coordinates.
    :return: list of mirrored patch crops: lenght=3. each element is a list of transformed patch crops.
    """
    mirrored_patch_crops = []

    # y-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3], ii[4], ii[5]] for ii in patch_crops])

    # x-axis transform.
    mirrored_patch_crops.append([[ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    # y-axis and x-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    return mirrored_patch_crops




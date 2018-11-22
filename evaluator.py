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
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import utils.model_utils as mutils
import plotting
from multiprocessing import Pool


class Evaluator():


    def __init__(self, cf, logger, mode='test'):
        """
        :param mode: either 'val_sampling', 'val_patient' or 'test'. handles prediction lists of different forms.
        """
        self.cf = cf
        self.logger = logger
        self.mode = mode


    def evaluate_predictions(self, results_list, monitor_metrics=None):
        """
        Performs the matching of predicted boxes and ground truth boxes. Loops over list of matching IoUs and foreground classes.
        Resulting info of each prediction is stored as one line in an internal dataframe, with the keys:
        det_type: 'tp' (true positive), 'fp' (false positive), 'fn' (false negative), 'tn' (true negative)
        pred_class: foreground class which the object predicts.
        pid: corresponding patient-id.
        pred_score: confidence score [0, 1]
        fold: corresponding fold of CV.
        match_iou: utilized IoU for matching.
        :param results_list: list of model predictions. Either from train/val_sampling (patch processing) for monitoring with form:
        [[[results_0, ...], [pid_0, ...]], [[results_n, ...], [pid_n, ...]], ...]
        Or from val_patient/testing (patient processing), with form: [[results_0, pid_0], [results_1, pid_1], ...])
        :param monitor_metrics (optional):  dict of dicts with all metrics of previous epochs.
        :return monitor_metrics: if provided (during training), return monitor_metrics now including results of current epoch.
        """
        # gets results_list = [[batch_instances_box_lists], [batch_instances_pids]]*n_batches
        # we want to evaluate one batch_instance (= 2D or 3D image) at a time.

        df_list_preds = []
        df_list_labels = []
        df_list_class_preds = []
        df_list_pids = []
        df_list_type = []
        df_list_match_iou = []

        self.logger.info('evaluating in mode {}'.format(self.mode))


        if self.mode == 'train' or self.mode=='val_sampling':
            # batch_size > 1, with varying patients across batch:
            # [[[results_0, ...], [pid_0, ...]], [[results_n, ...], [pid_n, ...]], ...]
            # -> [results_0, results_1, ..] , [pid_0, pid_1, ...]
            batch_elements_list = [[b_box_list] for item in results_list for b_box_list in item[0]]
            pid_list = [pid for item in results_list for pid in item[1]]
        else:
            # patient processing, one element per batch = one patient.
            # [[results_0, pid_0], [results_1, pid_1], ...] -> [results_0, results_1, ..] , [pid_0, pid_1, ...]
            batch_elements_list = [item[0] for item in results_list]
            pid_list = [item[1] for item in results_list]

        for match_iou in self.cf.ap_match_ious:
            self.logger.info('evaluating with match_iou: {}'.format(match_iou))
            for cl in list(self.cf.class_dict.keys()):
                for pix, pid in enumerate(pid_list):

                    len_df_list_before_patient = len(df_list_pids)

                    # input of each batch element is a list of boxes, where each box is a dictionary.
                    for bix, b_boxes_list in enumerate(batch_elements_list[pix]):

                        b_tar_boxes = np.array([box['box_coords'] for box in b_boxes_list if
                                                (box['box_type'] == 'gt' and box['box_label'] == cl)])
                        b_cand_boxes = np.array([box['box_coords'] for box in b_boxes_list if
                                                 (box['box_type'] == 'det' and
                                                  box['box_pred_class_id'] == cl)])
                        b_cand_scores = np.array([box['box_score'] for box in b_boxes_list if
                                                  (box['box_type'] == 'det' and
                                                   box['box_pred_class_id'] == cl)])

                        # check if predictions and ground truth boxes exist and match them according to match_iou.
                        if not 0 in b_cand_boxes.shape and not 0 in b_tar_boxes.shape:
                            overlaps = mutils.compute_overlaps(b_cand_boxes, b_tar_boxes)
                            match_cand_ixs = np.argwhere(np.max(overlaps, 1) > match_iou)[:, 0]
                            non_match_cand_ixs = np.argwhere(np.max(overlaps, 1) <= match_iou)[:, 0]
                            match_gt_ixs = np.argmax(overlaps[match_cand_ixs, :],
                                                     1) if not 0 in match_cand_ixs.shape else np.array([])
                            non_match_gt_ixs = np.array(
                                [ii for ii in np.arange(b_tar_boxes.shape[0]) if ii not in match_gt_ixs])
                            unique, counts = np.unique(match_gt_ixs, return_counts=True)

                            # check for double assignments, i.e. two predictions having been assigned to the same gt.
                            # according to the COCO-metrics, only one prediction counts as true positive, the rest counts as
                            # false positive. This case is supposed to be avoided by the model itself by,
                            #  e.g. using a low enough NMS threshold.
                            if np.any(counts > 1):
                                double_match_gt_ixs = unique[np.argwhere(counts > 1)[:, 0]]
                                keep_max = []
                                double_match_list = []
                                for dg in double_match_gt_ixs:
                                    double_match_cand_ixs = match_cand_ixs[np.argwhere(match_gt_ixs == dg)]
                                    keep_max.append(double_match_cand_ixs[np.argmax(b_cand_scores[double_match_cand_ixs])])
                                    double_match_list += [ii for ii in double_match_cand_ixs]

                                fp_ixs = np.array([ii for ii in match_cand_ixs if
                                                     (ii in double_match_list and ii not in keep_max)])

                                match_cand_ixs = np.array([ii for ii in match_cand_ixs if ii not in fp_ixs])

                                df_list_preds += [ii for ii in b_cand_scores[fp_ixs]]
                                df_list_labels += [0] * fp_ixs.shape[0]
                                df_list_class_preds += [cl] * fp_ixs.shape[0]
                                df_list_pids += [pid] * fp_ixs.shape[0]
                                df_list_type += ['det_fp'] * fp_ixs.shape[0]

                            # matched:
                            if not 0 in match_cand_ixs.shape:
                                df_list_preds += [ii for ii in b_cand_scores[match_cand_ixs]]
                                df_list_labels += [1] * match_cand_ixs.shape[0]
                                df_list_class_preds += [cl] * match_cand_ixs.shape[0]
                                df_list_pids += [pid] * match_cand_ixs.shape[0]
                                df_list_type += ['det_tp'] * match_cand_ixs.shape[0]
                            # rest fp:
                            if not 0 in non_match_cand_ixs.shape:
                                df_list_preds += [ii for ii in b_cand_scores[non_match_cand_ixs]]
                                df_list_labels += [0] * non_match_cand_ixs.shape[0]
                                df_list_class_preds += [cl] * non_match_cand_ixs.shape[0]
                                df_list_pids += [pid] * non_match_cand_ixs.shape[0]
                                df_list_type += ['det_fp'] * non_match_cand_ixs.shape[0]
                            # rest fn:
                            if not 0 in non_match_gt_ixs.shape:
                                df_list_preds += [0] * non_match_gt_ixs.shape[0]
                                df_list_labels += [1] * non_match_gt_ixs.shape[0]
                                df_list_class_preds += [cl] * non_match_gt_ixs.shape[0]
                                df_list_pids += [pid]  * non_match_gt_ixs.shape[0]
                                df_list_type += ['det_fn']  * non_match_gt_ixs.shape[0]
                        # only fp:
                        if not 0 in b_cand_boxes.shape and 0 in b_tar_boxes.shape:
                            df_list_preds += [ii for ii in b_cand_scores]
                            df_list_labels += [0] * b_cand_scores.shape[0]
                            df_list_class_preds += [cl] * b_cand_scores.shape[0]
                            df_list_pids += [pid] * b_cand_scores.shape[0]
                            df_list_type += ['det_fp'] * b_cand_scores.shape[0]
                        # only fn:
                        if 0 in b_cand_boxes.shape and not 0 in b_tar_boxes.shape:
                            df_list_preds += [0] * b_tar_boxes.shape[0]
                            df_list_labels += [1] * b_tar_boxes.shape[0]
                            df_list_class_preds += [cl] * b_tar_boxes.shape[0]
                            df_list_pids += [pid] * b_tar_boxes.shape[0]
                            df_list_type += ['det_fn'] * b_tar_boxes.shape[0]

                    # empty patient with 0 detections needs patient dummy score, in order to not disappear from stats.
                    # filtered out for roi-level evaluation later. During training (and val_sampling),
                    # tn are assigned per sample independently of associated patients.
                    if len(df_list_pids) == len_df_list_before_patient:
                        df_list_preds += [0] * 1
                        df_list_labels += [0] * 1
                        df_list_class_preds += [cl] * 1
                        df_list_pids += [pid] * 1
                        df_list_type += ['patient_tn'] * 1 # true negative: no ground truth boxes, no detections.

            df_list_match_iou += [match_iou] * (len(df_list_preds) - len(df_list_match_iou))

        self.test_df = pd.DataFrame()
        self.test_df['pred_score'] = df_list_preds
        self.test_df['class_label'] = df_list_labels
        self.test_df['pred_class'] = df_list_class_preds
        self.test_df['pid'] = df_list_pids
        self.test_df['det_type'] = df_list_type
        self.test_df['fold'] = self.cf.fold
        self.test_df['match_iou'] = df_list_match_iou
        if monitor_metrics is not None:
            return self.return_metrics(monitor_metrics)


    def return_metrics(self, monitor_metrics=None):
        """
        calculates AP/AUC scores for internal dataframe. called directly from evaluate_predictions during training for monitoring,
        or from score_test_df during inference (for single folds or aggregated test set). Loops over foreground classes
        and score_levels (typically 'roi' and 'patient'), gets scores and stores them. Optionally creates plots of
        prediction histograms and roc/prc curves.
        :param monitor_metrics: dict of dicts with all metrics of previous epochs.
        this function adds metrics for current epoch and returns the same object.
        :return: all_stats: list. Contains dicts with resulting scores for each combination of foreground class and
        score_level.
        :return: monitor_metrics
        """
        df = self.test_df

        all_stats = []
        for cl in list(self.cf.class_dict.keys()):
            cl_df = df[df.pred_class == cl]

            for score_level in self.cf.report_score_level:
                stats_dict = {}
                stats_dict['name'] = 'fold_{} {} cl_{}'.format(self.cf.fold, score_level, cl)

                if score_level == 'rois':
                    # kick out dummy entries for true negative patients. not needed on roi-level.
                    spec_df = cl_df[cl_df.det_type != 'patient_tn']
                    stats_dict['ap'] = get_roi_ap_from_df([spec_df, self.cf.min_det_thresh, self.cf.per_patient_ap])
                    # AUC not sensible on roi-level, since true negative box predictions do not exist. Would reward
                    # higher amounts of low confidence false positives.
                    stats_dict['auc'] = 0
                    stats_dict['roc'] = None
                    stats_dict['prc'] = None

                    # for the aggregated test set case, additionally get the scores for averaging over fold results.
                    if len(df.fold.unique()) > 1:
                        aps = []
                        for fold in df.fold.unique():
                            fold_df = spec_df[spec_df.fold == fold]
                            aps.append(get_roi_ap_from_df([fold_df, self.cf.min_det_thresh, self.cf.per_patient_ap]))
                        stats_dict['mean_ap'] = np.mean(aps)
                        stats_dict['mean_auc'] = 0

                # on patient level, aggregate predictions per patient (pid): The patient predicted score is the highest
                # confidence prediction for this class. The patient class label is 1 if roi of this class exists in patient, else 0.
                if score_level == 'patient':
                    spec_df = cl_df.groupby(['pid'], as_index=False).agg({'class_label': 'max', 'pred_score': 'max', 'fold': 'first'})

                    if len(spec_df.class_label.unique()) > 1:
                        stats_dict['auc'] = roc_auc_score(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                        stats_dict['roc'] = roc_curve(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                    else:
                        stats_dict['auc'] = np.nan
                        stats_dict['roc'] = np.nan

                    if (spec_df.class_label == 1).any():
                        stats_dict['ap'] = average_precision_score(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                        stats_dict['prc'] = precision_recall_curve(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                    else:
                        stats_dict['ap'] = np.nan
                        stats_dict['prc'] = np.nan

                    # for the aggregated test set case, additionally get the scores for averaging over fold results.
                    if len(df.fold.unique()) > 1:
                        aucs = []
                        aps = []
                        for fold in df.fold.unique():
                            fold_df = spec_df[spec_df.fold == fold]
                            if len(fold_df.class_label.unique()) > 1:
                                aucs.append(roc_auc_score(fold_df.class_label.tolist(), fold_df.pred_score.tolist()))
                            if (fold_df.class_label == 1).any():
                                aps.append(average_precision_score(fold_df.class_label.tolist(), fold_df.pred_score.tolist()))
                        stats_dict['mean_auc'] = np.mean(aucs)
                        stats_dict['mean_ap'] = np.mean(aps)

                # fill new results into monitor_metrics dict. for simplicity, only one class (of interest) is monitored on patient level.
                if monitor_metrics is not None and not (score_level == 'patient' and cl != self.cf.patient_class_of_interest):
                    score_level_name = 'patient' if score_level == 'patient' else self.cf.class_dict[cl]
                    monitor_metrics[score_level_name + '_ap'].append(stats_dict['ap'] if stats_dict['ap'] > 0 else None)
                    if score_level == 'patient':
                        monitor_metrics[score_level_name + '_auc'].append(
                            stats_dict['auc'] if stats_dict['auc'] > 0 else None)

                if self.cf.plot_prediction_histograms:
                    out_filename = os.path.join(
                        self.cf.plot_dir, 'pred_hist_{}_{}_{}_cl{}'.format(
                            self.cf.fold, 'val' if 'val' in self.mode else self.mode, score_level, cl))
                    type_list = None if score_level == 'patient' else spec_df.det_type.tolist()
                    plotting.plot_prediction_hist(spec_df.class_label.tolist(), spec_df.pred_score.tolist(), type_list, out_filename)

                all_stats.append(stats_dict)

                # analysis of the  hyper-parameter cf.min_det_thresh, for optimization on validation set.
                if self.cf.scan_det_thresh:
                    conf_threshs = list(np.arange(0.9, 1, 0.01))
                    pool = Pool(processes=10)
                    mp_inputs = [[spec_df, ii, self.cf.per_patient_ap] for ii in conf_threshs]
                    aps = pool.map(get_roi_ap_from_df, mp_inputs, chunksize=1)
                    pool.close()
                    pool.join()
                    self.logger.info('results from scanning over det_threshs:', [[i, j] for i, j in zip(conf_threshs, aps)])

        if self.cf.plot_stat_curves:
            out_filename = os.path.join(self.cf.plot_dir, '{}_{}_stat_curves'.format(self.cf.fold, self.mode))
            plotting.plot_stat_curves(all_stats, out_filename)


        # get average stats over foreground classes on roi level.
        avg_ap = np.mean([d['ap'] for d in all_stats if 'rois' in d['name']])
        all_stats.append({'name': 'average_foreground_roi', 'auc': 0, 'ap': avg_ap})
        if len(df.fold.unique()) > 1:
            avg_mean_ap = np.mean([d['mean_ap'] for d in all_stats if 'rois' in d['name']])
            all_stats[-1]['mean_ap'] = avg_mean_ap
            all_stats[-1]['mean_auc'] = 0

        # in small data sets, values of model_selection_criterion can be identical across epochs, wich breaks the
        # ranking of model_selector. Thus, pertube identical values by a neglectibale random term.
        for sc in self.cf.model_selection_criteria:
            if 'val' in self.mode and monitor_metrics[sc].count(monitor_metrics[sc][-1]) > 1 and monitor_metrics[sc][-1] is not None:
                monitor_metrics[sc][-1] += 1e-6 * np.random.rand()

        return all_stats, monitor_metrics


    def score_test_df(self, internal_df=True):
        """
        Writes out resulting scores to text files: First checks for class-internal-df (typically current) fold,
        gets resulting scores, writes them to a text file and pickles data frame. Also checks if data-frame pickles of
        all folds of cross-validation exist in exp_dir. If true, loads all dataframes, aggregates test sets over folds,
        and calculates and writes out overall metrics.
        """
        if internal_df:

            self.test_df.to_pickle(os.path.join(self.cf.exp_dir, '{}_test_df.pickle'.format(self.cf.fold)))
            stats, _ = self.return_metrics()

            with open(os.path.join(self.cf.exp_dir, 'results.txt'), 'a') as handle:
                handle.write('\n****************************\n')
                handle.write('\nresults for fold {} \n'.format(self.cf.fold))
                handle.write('\n****************************\n')
                handle.write('\nfold df shape {}\n  \n'.format(self.test_df.shape))
                for s in stats:
                    handle.write('AUC {:0.4f}  AP {:0.4f} {} \n'.format(s['auc'], s['ap'], s['name']))

        fold_df_paths = [ii for ii in os.listdir(self.cf.exp_dir) if 'test_df.pickle' in ii]
        if len(fold_df_paths) == self.cf.n_cv_splits:
            with open(os.path.join(self.cf.exp_dir, 'results.txt'), 'a') as handle:
                self.cf.fold = 'overall'
                dfs_list = [pd.read_pickle(os.path.join(self.cf.exp_dir, ii)) for ii in fold_df_paths]
                for ix, df in enumerate(dfs_list):
                    df['fold'] = ix
                self.test_df = pd.concat(dfs_list)
                stats, _ = self.return_metrics()
                handle.write('\n****************************\n')
                handle.write('\nOVERALL RESULTS \n')
                handle.write('\n****************************\n')
                handle.write('\ndf shape \n  \n'.format(self.test_df.shape))
                for s in stats:
                    handle.write('\nAUC {:0.4f} (mu {:0.4f})  AP {:0.4f} (mu {:0.4f})  {}\n '
                                 .format(s['auc'], s['mean_auc'], s['ap'], s['mean_ap'], s['name']))
                results_table_path = os.path.join(("/").join(self.cf.exp_dir.split("/")[:-1]), 'results_table.txt')
                with open(results_table_path, 'a') as handle2:
                    for s in stats:
                        handle2.write('\nAUC {:0.4f} (mu {:0.4f})  AP {:0.4f} (mu {:0.4f})  {} {}'
                                      .format(s['auc'], s['mean_auc'], s['ap'], s['mean_ap'], s['name'], self.cf.exp_dir.split('/')[-1]))
                    handle2.write('\n')



def get_roi_ap_from_df(inputs):
    '''
    :param df: data frame.
    :param det_thresh: min_threshold for filtering out low confidence predictions.
    :param per_patient_ap: boolean flag. evaluate average precision per image and average over images,
    instead of computing one ap over data set.
    :return: average_precision (float)
    '''
    df, det_thresh, per_patient_ap = inputs

    if per_patient_ap:
        pids_list = df.pid.unique()
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            for pid in pids_list:
                pid_df = iou_df[iou_df.pid == pid]
                all_p = len(pid_df[pid_df.class_label == 1])
                pid_df = pid_df[(pid_df.det_type == 'det_fp') | (pid_df.det_type == 'det_tp')].sort_values('pred_score', ascending=False)
                pid_df = pid_df[pid_df.pred_score > det_thresh]
                if (len(pid_df) ==0 and all_p == 0):
                   pass
                elif (len(pid_df) > 0 and all_p == 0):
                    aps.append(0)
                else:
                    aps.append(compute_roi_ap(pid_df, all_p))
        return np.mean(aps)

    else:
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            all_p = len(iou_df[iou_df.class_label == 1])
            iou_df = iou_df[(iou_df.det_type == 'det_fp') | (iou_df.det_type == 'det_tp')].sort_values('pred_score', ascending=False)
            iou_df = iou_df[iou_df.pred_score > det_thresh]
            if all_p > 0:
                aps.append(compute_roi_ap(iou_df, all_p))
        return np.mean(aps)



def compute_roi_ap(df, all_p):
    """
    adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    :param df: dataframe containing class labels of predictions sorted in descending manner by their prediction score.
    :param all_p: number of all ground truth objects. (for denominator of recall.)
    :return:
    """
    tp = df.class_label.values
    fp = (tp == 0) * 1
    #recall thresholds, where precision will be measured
    R = np.linspace(.0, 1, 101, endpoint=True)
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(fp)
    nd = len(tp)
    rc = tp_sum / all_p
    pr = tp_sum / (fp_sum + tp_sum)
    # initialize precision array over recall steps.
    q = np.zeros((len(R),))

    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist()
    q = q.tolist()
    for i in range(nd - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    #discretize empiric recall steps with given bins.
    inds = np.searchsorted(rc, R, side='left')
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
    except:
        pass

    return np.mean(q)
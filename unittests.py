#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
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

import unittest

import os
import pickle
import time
from multiprocessing import  Pool
import subprocess

import numpy as np
import pandas as pd
import torch
import torchvision as tv

import tqdm

import utils.exp_utils as utils
import utils.model_utils as mutils

""" Note on unittests: run this file either in the way intended for unittests by starting the script with
    python -m unittest unittests.py or start it as a normal python file as python unittests.py.
    You can selective run single tests by calling python -m unittest unittests.TestClassOfYourChoice, where 
    TestClassOfYourChoice is the name of the test defined below, e.g., CompareFoldSplits.
"""



def inspect_info_df(pp_dir):
    """ use your debugger to look into the info df of a pp dir.
    :param pp_dir: preprocessed-data directory
    """

    info_df = pd.read_pickle(os.path.join(pp_dir, "info_df.pickle"))

    return


def generate_boxes(count, dim=2, h=100, w=100, d=20, normalize=False, on_grid=False, seed=0):
    """ generate boxes of format [y1, x1, y2, x2, (z1, z2)].
    :param count: nr of boxes
    :param dim: dimension of boxes (2 or 3)
    :return: boxes in format (n_boxes, 4 or 6), scores
    """
    np.random.seed(seed)
    if on_grid:
        lower_y = np.random.randint(0, h // 2, (count,))
        lower_x = np.random.randint(0, w // 2, (count,))
        upper_y = np.random.randint(h // 2, h, (count,))
        upper_x = np.random.randint(w // 2, w, (count,))
        if dim == 3:
            lower_z = np.random.randint(0, d // 2, (count,))
            upper_z = np.random.randint(d // 2, d, (count,))
    else:
        lower_y = np.random.rand(count) * h / 2.
        lower_x = np.random.rand(count) * w / 2.
        upper_y = (np.random.rand(count) + 1.) * h / 2.
        upper_x = (np.random.rand(count) + 1.) * w / 2.
        if dim == 3:
            lower_z = np.random.rand(count) * d / 2.
            upper_z = (np.random.rand(count) + 1.) * d / 2.

    if dim == 3:
        boxes = np.array(list(zip(lower_y, lower_x, upper_y, upper_x, lower_z, upper_z)))
        # add an extreme box that tests the boundaries
        boxes = np.concatenate((boxes, np.array([[0., 0., h, w, 0, d]])))
    else:
        boxes = np.array(list(zip(lower_y, lower_x, upper_y, upper_x)))
        boxes = np.concatenate((boxes, np.array([[0., 0., h, w]])))

    scores = np.random.rand(count + 1)
    if normalize:
        divisor = np.array([h, w, h, w, d, d]) if dim == 3 else np.array([h, w, h, w])
        boxes = boxes / divisor
    return boxes, scores


# -------- check own nms CUDA implement against own numpy implement ------
class CheckNMSImplementation(unittest.TestCase):

    @staticmethod
    def assert_res_equality(keep_ics1, keep_ics2, boxes, scores, tolerance=0, names=("res1", "res2")):
        """
        :param keep_ics1: keep indices (results), torch.Tensor of shape (n_ics,)
        :param keep_ics2:
        :return:
        """
        keep_ics1, keep_ics2 = keep_ics1.cpu().numpy(), keep_ics2.cpu().numpy()
        discrepancies = np.setdiff1d(keep_ics1, keep_ics2)
        try:
            checks = np.array([
                len(discrepancies) <= tolerance
            ])
        except:
            checks = np.zeros((1,)).astype("bool")
        msgs = np.array([
            """{}: {} \n{}: {} \nboxes: {}\n {}\n""".format(names[0], keep_ics1, names[1], keep_ics2, boxes,
                                                            scores)
        ])

        assert np.all(checks), "NMS: results mismatch: " + "\n".join(msgs[~checks])

    def single_case(self, count=20, dim=3, threshold=0.2, seed=0):
        boxes, scores = generate_boxes(count, dim, seed=seed, h=320, w=280, d=30)

        keep_numpy = torch.tensor(mutils.nms_numpy(boxes, scores, threshold))

        # for some reason torchvision nms requires box coords as floats.
        boxes = torch.from_numpy(boxes).type(torch.float32)
        scores = torch.from_numpy(scores).type(torch.float32)
        if dim == 2:
            """need to wait until next pytorch release where they fixed nms on cpu (currently they have >= where it
            needs to be >.)
            """
            # keep_ops = tv.ops.nms(boxes, scores, threshold)
            # self.assert_res_equality(keep_numpy, keep_ops, boxes, scores, tolerance=0, names=["np", "ops"])
            pass

        boxes = boxes.cuda()
        scores = scores.cuda()
        keep = self.nms_ext.nms(boxes, scores, threshold)
        self.assert_res_equality(keep_numpy, keep, boxes, scores, tolerance=0, names=["np", "cuda"])

    def manual_example(self):
        """
        100 x 221 (y, x) image. 5 overlapping boxes, 4 of the same class, 3 of them overlapping above threshold.

        """
        threshold = 0.3
        boxes = torch.tensor([
            [20, 30, 80, 130], #0 reference (needs to have highest score)
            [30, 40, 70, 120], #1 IoU 0.35
            [10, 50, 90,  80], #2 IoU 0.11
            [40, 20, 75, 135], #3 IoU 0.34
            [30, 40, 70, 120], #4 IoU 0.35 again but with lower score
        ]).cuda().float()

        scores = torch.tensor([0.71, 0.94, 1.0, 0.82, 0.11]).cuda()

        # expected: keep == [1, 2]
        keep = self.nms_ext.nms(boxes, scores, threshold)

        diff = np.setdiff1d(keep.cpu().numpy(), [1,2])
        assert len(diff) == 0, "expected: {}, received: {}.".format([1,2], keep)



    def test(self, n_cases=200, box_count=30, threshold=0.5):
        # dynamically import module so that it doesn't affect other tests if import fails
        self.nms_ext = utils.import_module("nms_ext", 'custom_extensions/nms/nms.py')

        self.manual_example()

        # change seed to something fix if you want exactly reproducible test
        seed0 = np.random.randint(50)
        print("NMS test progress (done/total box configurations) 2D:", end="\n")
        for i in tqdm.tqdm(range(n_cases)):
            self.single_case(count=box_count, dim=2, threshold=threshold, seed=seed0+i)
        print("NMS test progress (done/total box configurations) 3D:", end="\n")
        for i in tqdm.tqdm(range(n_cases)):
            self.single_case(count=box_count, dim=3, threshold=threshold, seed=seed0+i)

        return

class CheckRoIAlignImplementation(unittest.TestCase):

    def prepare(self, dim=2):

        b, c, h, w = 1, 3, 50, 50
        # feature map, (b, c, h, w(, z))
        if dim == 2:
            fmap = torch.rand(b, c, h, w).cuda()
            # rois = torch.tensor([[
            #     [0.1, 0.1, 0.3, 0.3],
            #     [0.2, 0.2, 0.4, 0.7],
            #     [0.5, 0.7, 0.7, 0.9],
            # ]]).cuda()
            pool_size = (7, 7)
            rois = generate_boxes(5, dim=dim, h=h, w=w, on_grid=True, seed=np.random.randint(50))[0]
        elif dim == 3:
            d = 20
            fmap = torch.rand(b, c, h, w, d).cuda()
            # rois = torch.tensor([[
            #     [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
            #     [0.2, 0.2, 0.4, 0.7, 0.2, 0.4],
            #     [0.5, 0.0, 0.7, 1.0, 0.4, 0.5],
            #     [0.0, 0.0, 0.9, 1.0, 0.0, 1.0],
            # ]]).cuda()
            pool_size = (7, 7, 3)
            rois = generate_boxes(5, dim=dim, h=h, w=w, d=d, on_grid=True, seed=np.random.randint(50),
                                  normalize=False)[0]
        else:
            raise ValueError("dim needs to be 2 or 3")

        rois = [torch.from_numpy(rois).type(dtype=torch.float32).cuda(), ]
        fmap.requires_grad_(True)
        return fmap, rois, pool_size

    def check_2d(self):
        """ check vs torchvision ops not possible as on purpose different approach.
        :return:
        """
        raise NotImplementedError
        # fmap, rois, pool_size = self.prepare(dim=2)
        # ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1., sampling_ratio=-1)
        # align_ext = ra_object(fmap, rois)
        # loss_ext = align_ext.sum()
        # loss_ext.backward()
        #
        # rois_swapped = [rois[0][:, [1,3,0,2]]]
        # align_ops = tv.ops.roi_align(fmap, rois_swapped, pool_size)
        # loss_ops = align_ops.sum()
        # loss_ops.backward()
        #
        # assert (loss_ops == loss_ext), "sum of roialign ops and extension 2D diverges"
        # assert (align_ops == align_ext).all(), "ROIAlign failed 2D test"

    def check_3d(self):
        fmap, rois, pool_size = self.prepare(dim=3)
        ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1., sampling_ratio=-1)
        align_ext = ra_object(fmap, rois)
        loss_ext = align_ext.sum()
        loss_ext.backward()

        align_np = mutils.roi_align_3d_numpy(fmap.cpu().detach().numpy(), [roi.cpu().numpy() for roi in rois],
                                             pool_size)
        align_np = np.squeeze(align_np)  # remove singleton batch dim

        align_ext = align_ext.cpu().detach().numpy()
        assert np.allclose(align_np, align_ext, rtol=1e-5,
                           atol=1e-8), "RoIAlign differences in numpy and CUDA implement"

    def specific_example_check(self):
        # dummy input
        self.ra_ext = utils.import_module("ra_ext", 'custom_extensions/roi_align/roi_align.py')
        exp = 6
        pool_size = (2,2)
        fmap = torch.arange(exp**2).view(exp,exp).unsqueeze(0).unsqueeze(0).cuda().type(dtype=torch.float32)

        boxes = torch.tensor([[1., 1., 5., 5.]]).cuda()/exp
        ind = torch.tensor([0.]*len(boxes)).cuda().type(torch.float32)
        y_exp, x_exp = fmap.shape[2:]  # exp = expansion
        boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
        boxes = torch.cat((ind.unsqueeze(1), boxes), dim=1)
        aligned_tv = tv.ops.roi_align(fmap, boxes, output_size=pool_size, sampling_ratio=-1)
        aligned = self.ra_ext.roi_align_2d(fmap, boxes, output_size=pool_size, sampling_ratio=-1)

        boxes_3d = torch.cat((boxes, torch.tensor([[-1.,1.]]*len(boxes)).cuda()), dim=1)
        fmap_3d = fmap.unsqueeze(dim=-1)
        pool_size = (*pool_size,1)
        ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1.,)
        aligned_3d = ra_object(fmap_3d, boxes_3d)

        # expected_res = torch.tensor([[[[10.5000, 12.5000], # this would be with an alternative grid-point setting
        #                                [22.5000, 24.5000]]]]).cuda()
        expected_res = torch.tensor([[[[14., 16.],
                                       [26., 28.]]]]).cuda()
        expected_res_3d = torch.tensor([[[[[14.],[16.]],
                                          [[26.],[28.]]]]]).cuda()
        assert torch.all(aligned==expected_res), "2D RoIAlign check vs. specific example failed. res: {}\n expected: {}\n".format(aligned, expected_res)
        assert torch.all(aligned_3d==expected_res_3d), "3D RoIAlign check vs. specific example failed. res: {}\n expected: {}\n".format(aligned_3d, expected_res_3d)


    def test(self):
        # dynamically import module so that it doesn't affect other tests if import fails
        self.ra_ext = utils.import_module("ra_ext", 'custom_extensions/roi_align/roi_align.py')

        self.specific_example_check()

        # 2d test
        #self.check_2d()

        # 3d test
        self.check_3d()

        return

class VerifyFoldSplits(unittest.TestCase):
    """ Check, for a single fold_ids file, i.e., for a single experiment, if the assigned folds (assignment of data
        identifiers) is actually incongruent. No overlaps between folds are allowed for a correct cross validation.
    """
    @staticmethod
    def verify_fold_ids(splits):
        """
        Splits: list (n_splits). Each element: list (4) with: 0 == array of train ids, 1 == arr of val ids,
        2 == arr of test ids, 3 == int of fold ix.
        """

        for f_ix, split_settings in enumerate(splits):
            split_ids, fold_ix = split_settings[:3], split_settings[3]
            assert f_ix == fold_ix

            # check fold ids within their folds
            for i, ids1 in enumerate(split_ids):
                for j, ids2 in enumerate(split_ids):
                    if j > i:
                        inter = np.intersect1d(ids1, ids2)
                        if len(inter) > 0:
                            raise Exception("Fold {}: Split {} and {} intersect by pids {}".format(fold_ix, i, j, inter))

            # check val and test ids across folds
            val_ids = split_ids[1]
            test_ids = split_ids[2]
            for other_f_ix in range(f_ix + 1, len(splits)):
                other_val_ids = splits[other_f_ix][1]
                other_test_ids = splits[other_f_ix][2]
                inter_val = np.intersect1d(val_ids, other_val_ids)
                inter_test = np.intersect1d(test_ids, other_test_ids)
                if len(inter_test) > 0:
                    raise Exception("Folds {} and {}: Test splits intersect by pids {}".format(f_ix, other_f_ix, inter_test))
                if len(inter_val) > 0:
                    raise Exception(
                        "Folds {} and {}: Val splits intersect by pids {}".format(f_ix, other_f_ix, inter_val))

    def test(self):
        exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/lidc_exp/experiments/042/retinau2d"
        check_file = os.path.join(exp_dir, 'fold_ids.pickle')
        with open(check_file, 'rb') as handle:
            splits = pickle.load(handle)
        self.verify_fold_ids(splits)


class CompareFoldSplits(unittest.TestCase):
    """ Find evtl. differences in cross-val file splits across different experiments.
    """
    @staticmethod
    def group_id_paths(ref_exp_dir, comp_exp_dirs):

        f_name = 'fold_ids.pickle'

        ref_paths = os.path.join(ref_exp_dir, f_name)
        assert os.path.isfile(ref_paths), "ref file {} does not exist.".format(ref_paths)


        ref_paths = [ref_paths for comp_ed in comp_exp_dirs]
        comp_paths = [os.path.join(comp_ed, f_name) for comp_ed in comp_exp_dirs]

        return zip(ref_paths, comp_paths)

    @staticmethod
    def comp_fold_ids(mp_input):
        fold_ids1, fold_ids2 = mp_input
        with open(fold_ids1, 'rb') as f:
            fold_ids1 = pickle.load(f)
        try:
            with open(fold_ids2, 'rb') as f:
                fold_ids2 = pickle.load(f)
        except FileNotFoundError:
            print("comp file {} does not exist.".format(fold_ids2))
            return

        n_splits = len(fold_ids1)
        assert n_splits == len(fold_ids2), "mismatch n splits: ref has {}, comp {}".format(n_splits, len(fold_ids2))
        # train, val test
        split_diffs = np.concatenate([np.setdiff1d(fold_ids1[s][assignment], fold_ids2[s][assignment]) for s in range(n_splits) for assignment in range(3)])
        all_equal = np.any(split_diffs)
        return (split_diffs, all_equal)

    def iterate_exp_dirs(self, ref_exp, comp_exps, processes=os.cpu_count()):

        grouped_paths = list(self.group_id_paths(ref_exp, comp_exps))
        print("performing {} comparisons of cross-val file splits".format(len(grouped_paths)))
        p = Pool(processes)
        split_diffs = p.map(self.comp_fold_ids, grouped_paths)
        p.close(); p.join()

        df = pd.DataFrame(index=range(0,len(grouped_paths)), columns=["ref", "comp", "all_equal"])#, "diffs"])
        for ix, (ref, comp) in enumerate(grouped_paths):
            df.iloc[ix] = [ref, comp, split_diffs[ix][1]]#, split_diffs[ix][0]]

        print("Any splits not equal?", df.all_equal.any())
        assert not df.all_equal.any(), "a split set is different from reference split set, {}".format(df[~df.all_equal])

    def test(self):
        exp_parent_dir = '/home/gregor/networkdrives/E132-Cluster-Projects/lidc_exp/experiments/1x/adamw_nonorm_nosched'
        ref_exp = '/media/gregor/HDD1/experiments/mdt/lidc_exp/original_paper_settings'
        comp_exps = [os.path.join(exp_parent_dir, p) for p in os.listdir(exp_parent_dir)]
        comp_exps = [p for p in comp_exps if os.path.isdir(p) and p != ref_exp]
        self.iterate_exp_dirs(ref_exp, comp_exps)


if __name__=="__main__":
    stime = time.time()

    unittest.main()

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
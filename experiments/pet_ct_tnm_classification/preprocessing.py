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
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
from scipy.ndimage.measurements import label as lb
from scipy.ndimage.measurements import center_of_mass as com
import nrrd
from copy import deepcopy
from skimage.segmentation import clear_border

import configs
cf = configs.configs()

# if a rater did not identify a nodule, this vote counts as 0s on the pixels. and as 0 == background (or 1?) on the mal. score.
# will this lead to many surpressed nodules. yes. they are not stored in segmentation map and the mal. labels are discarded.
# a pixel counts as foreground, if at least 2 raters drew it as foreground.


def get_z_crops(x, ix, min_pix=1500, n_comps=2, rad_crit = 20000):
    final_slices = []

    for six in range(x.shape[0]):

        tx = np.copy(x[six]) < -600
        img_center = np.array(tx.shape) / 2
        tx = clear_border(tx)

        clusters, n_cands = lb(tx)
        count = np.unique(clusters, return_counts=True)
        keep_comps = np.array([ii for ii in np.argwhere((count[1] > min_pix)) if ii > 0]).astype(int)

        if len(keep_comps) > n_comps - 1:
            coms = com(tx, clusters, index=keep_comps)
            keep_com = [ix for ix, ii in enumerate(coms[0]) if
                        ((ii[0] - img_center[0]) ** 2 + (ii[1] - img_center[1]) ** 2 < rad_crit)]
            keep_comps = keep_comps[keep_com]

            if len(keep_comps) > n_comps - 1:
                final_slices.append(six)
                #      print('appending', six)

    z_min = np.min(final_slices) - 7
    z_max = np.max(final_slices) + 7
    dist = z_max - z_min
    if dist >= 151:
        print('trying again with min pix', min_pix + 500, rad_crit - 500,  ix, dist)
        z_min, z_max = get_z_crops(x, ix, min_pix=min_pix + 500, rad_crit= rad_crit - 500)
    if dist <= 43:
        print('trying again with one component', min_pix - 100, rad_crit + 100,  ix, dist)
        z_min, z_max = get_z_crops(x, ix, n_comps=1, min_pix=min_pix - 100, rad_crit = rad_crit + 100)

    print(z_min, z_max, z_max - z_min, ix)
    return z_min, z_max


def pp_patient(inputs):

    ix, path = inputs
    background_categories = ['M1b_brain', 'N_inflammation', 'T_benign', 'T_other']
    # for lix, l in enumerate(patient['class_target']):
    #     if l in background_categories:
    #         seg[seg == lix + 1] = 0
    #     else:
    #         seg[seg == lix + 1] = 1

    selection = [106, 273]

    if ix in selection:

        pid = ix
        print('processing', pid, path)
        x = sitk.ReadImage(os.path.join(path, 'lsa_ct.nii.gz'))
        p = sitk.ReadImage(os.path.join(path, 'lsa_pet.nii.gz'))
        readdata, header = nrrd.read(os.path.join(path, 'lsa.seg.nrrd'))
        if len(readdata.shape) == 3:
            readdata = readdata[None]
            spacing = np.diagonal(header['space directions'])
        else:
            spacing = np.diagonal(header['space directions'][1:, :])

        origin = header['space origin'] * np.sign(spacing)
        labels = [header[k].split('=')[-1] for k in header.keys() if '_Name' in k]
        seg = np.zeros_like(readdata[0])
        print('READDATA SHAPE', readdata.shape)
        for ix in range(readdata.shape[0]):
            if labels[ix] not in background_categories:
                seg[readdata[ix] == 1] = ix = 1

        seg = seg.astype('uint8')
        s = sitk.GetImageFromArray(np.transpose(seg, axes=(2, 1, 0)))
        s.SetSpacing(abs(spacing))
        s.SetOrigin(origin)

        x_spacing = x.GetSpacing()
        if x_spacing[0] < 0.95 or x_spacing[2] < 3:
            new_spacing = (0.976562, 0.976562, 3.27)
            new_size = [int(x.GetSize()[ii] * x_spacing[ii] / new_spacing[ii]) for ii in range(3)]
            reference_image = sitk.Image(new_size, x.GetPixelIDValue())
            reference_image.SetOrigin(x.GetOrigin())
            reference_image.SetDirection(x.GetDirection())
            reference_image.SetSpacing(new_spacing)

            # Resample without any smoothing.
            x = sitk.Resample(x, reference_image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(x)
        # §resampler.SetInterpolator()  # default linear
        rp = resampler.Execute(p)
        rs = resampler.Execute(s)
        pi = sitk.GetArrayFromImage(rp)
        si = sitk.GetArrayFromImage(rs)
        xi = sitk.GetArrayFromImage(x)

        zmin, zmax = get_z_crops(xi, ix)

        x = xi[zmin:zmax]
        p = pi[zmin:zmax]
        s = si[zmin:zmax]

        x = np.clip(x, -1200, 600)
        x = (1200 + x) / (600 + 1200)   # (x-a) / (b-a) * (c-d)
        x = (x - np.mean(x)) / np.std(x)

        # p = np.clip(p, 0, 2)
        #   p = (p) / (2)
        p = (p - np.mean(p)) / np.std(p)


        assert np.all(np.array(x.shape) == np.array(s.shape))

        img = np.concatenate((x[None], p[None])).astype(np.float32)

        remaining_comps = np.unique(s)
        remaining_labels = [ii for ix, ii in enumerate(labels) if ix + 1 in remaining_comps]
        s[s > 0] = 1

        fg_slices = [ii for ii in np.unique(np.argwhere(s != 0)[:, 0])]

        out_df = pd.read_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))
        out_df.loc[len(out_df)] = {'pid': pid, 'raw_pid': path.split('/')[-1], 'class_target': remaining_labels, 'fg_slices': fg_slices}
        out_df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))

        np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), s)
        np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), img)


def collectPaths(in_dir):

    paths = []
    for path, dirs, files in os.walk(in_dir):
        pet_files = [f for f in files if 'lsa_pet' in f]
        if len(files) > 0 and 'TNM' in path and len(pet_files) > 0:
            paths.append(path)

    return paths

if __name__ == "__main__":

    paths = collectPaths(cf.raw_data_dir)
    print('all paths', len(paths))


    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)
        # df = pd.DataFrame(columns=['pid', 'raw_pid', 'class_target', 'fg_slices'])
        # df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))

    pool = Pool(processes=8)
    p1 = pool.map(pp_patient, enumerate(paths), chunksize=1)
    pool.close()
    pool.join()
    # for i in enumerate(paths):
    #     pp_patient(i)
    #
    # subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)
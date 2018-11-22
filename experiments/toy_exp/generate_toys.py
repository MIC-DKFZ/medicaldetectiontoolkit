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
from multiprocessing import Pool
import configs as cf

def multi_processing_create_image(inputs):


    out_dir, six, foreground_margin, class_diameters, mode = inputs
    print('proceesing {} {}'.format(out_dir, six))

    img = np.random.rand(320, 320)
    seg = np.zeros((320, 320)).astype('uint8')
    center_x = np.random.randint(foreground_margin, img.shape[0] - foreground_margin)
    center_y = np.random.randint(foreground_margin, img.shape[1] - foreground_margin)
    class_id = np.random.randint(0, 2)

    for y in range(img.shape[0]):
        for x in range(img.shape[0]):
            if ((x - center_x) ** 2 + (y - center_y) ** 2 - class_diameters[class_id] ** 2) < 0:
                img[y][x] += 0.2
                seg[y][x] = 1

    if 'donuts' in mode:
        whole_diameter = 4
        if class_id == 1:
            for y in range(img.shape[0]):
                for x in range(img.shape[0]):
                    if ((x - center_x) ** 2 + (y - center_y) ** 2 - whole_diameter ** 2) < 0:
                        img[y][x] -= 0.2
                        if mode == 'donuts_shape':
                            seg[y][x] = 0

    out = np.concatenate((img[None], seg[None]))
    out_path = os.path.join(out_dir, '{}.npy'.format(six))
    df = pd.read_pickle(os.path.join(out_dir, 'info_df.pickle'))
    df.loc[len(df)] = [out_path, class_id, str(six)]
    df.to_pickle(os.path.join(out_dir, 'info_df.pickle'))
    np.save(out_path, out)


def get_toy_image_info(mode, n_images, out_dir, class_diameters=(20, 20)):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # enforced distance between object center and image edge.
    foreground_margin = np.max(class_diameters) // 2

    df = pd.DataFrame(columns=['path', 'class_id', 'pid'])
    df.to_pickle(os.path.join(out_dir, 'info_df.pickle'))
    return [[out_dir, six, foreground_margin, class_diameters, mode] for six in range(n_images)]


if __name__ == '__main__':

    cf = cf.configs()

    root_dir = os.path.join(cf.root_dir, 'donuts_shape')
    info = []
    info += get_toy_image_info(mode='donuts_shape', n_images=1500, out_dir=os.path.join(root_dir, 'train'))
    info += get_toy_image_info(mode='donuts_shape', n_images=1000, out_dir=os.path.join(root_dir, 'test'))

    root_dir = os.path.join(cf.root_dir, 'donuts_pattern')
    info += get_toy_image_info(mode='donuts_pattern', n_images=1500, out_dir=os.path.join(root_dir, 'train'))
    info += get_toy_image_info(mode='donuts_pattern', n_images=1000, out_dir=os.path.join(root_dir, 'test'))

    root_dir = os.path.join(cf.root_dir, 'circles_scale')
    info += get_toy_image_info(mode='circles_scale', n_images=1500, out_dir=os.path.join(root_dir, 'train'), class_diameters=(19, 20))
    info += get_toy_image_info(mode='circles_scale', n_images=1000, out_dir=os.path.join(root_dir, 'test'), class_diameters=(19, 20))

    print('starting creating {} images'.format(len(info)))
    pool = Pool(processes=12)
    pool.map(multi_processing_create_image, info, chunksize=1)
    pool.close()
    pool.join()


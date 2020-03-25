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
import pickle
from multiprocessing import Pool
import configs as cf

def multi_processing_create_image(inputs):


    out_dir, six, foreground_margin, class_diameters, mode = inputs
    print('processing {} {}'.format(out_dir, six))

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
    np.save(out_path, out)

    with open(os.path.join(out_dir, 'meta_info_{}.pickle'.format(six)), 'wb') as handle:
        pickle.dump([out_path, class_id, str(six)], handle)


def generate_experiment(exp_name, n_train_images, n_test_images, mode, class_diameters=(20, 20)):

    train_dir = os.path.join(cf.root_dir, exp_name, 'train')
    test_dir = os.path.join(cf.root_dir, exp_name, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # enforced distance between object center and image edge.
    foreground_margin = np.max(class_diameters) // 2

    info = []
    info += [[train_dir, six, foreground_margin, class_diameters, mode] for six in range(n_train_images)]
    info += [[test_dir, six, foreground_margin, class_diameters, mode] for six in range(n_test_images)]

    print('starting creating {} images'.format(len(info)))
    pool = Pool(processes=12)
    pool.map(multi_processing_create_image, info, chunksize=1)
    pool.close()
    pool.join()

    aggregate_meta_info(train_dir)
    aggregate_meta_info(test_dir)


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['path', 'class_id', 'pid'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))


if __name__ == '__main__':

    cf = cf.configs()

    generate_experiment('donuts_shape', n_train_images=1500, n_test_images=1000, mode='donuts_shape')
    generate_experiment('donuts_pattern', n_train_images=1500, n_test_images=1000, mode='donuts_pattern')
    generate_experiment('circles_scale', n_train_images=1500, n_test_images=1000, mode='circles_scale', class_diameters=(19, 20))




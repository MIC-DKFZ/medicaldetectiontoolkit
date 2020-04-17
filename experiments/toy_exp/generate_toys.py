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

import os, time
import numpy as np
import pandas as pd
import pickle
import argparse
from multiprocessing import Pool

def multi_processing_create_image(inputs):


    out_dir, six, foreground_margin, class_diameters, mode, noisy_bg = inputs
    print('processing {} {}'.format(out_dir, six))

    img = np.random.rand(320, 320) if noisy_bg else np.zeros((320, 320))
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
        hole_diameter = 4
        if class_id == 1:
            for y in range(img.shape[0]):
                for x in range(img.shape[0]):
                    if ((x - center_x) ** 2 + (y - center_y) ** 2 - hole_diameter ** 2) < 0:
                        img[y][x] -= 0.2
                        if mode == 'donuts_shape':
                            seg[y][x] = 0

    out = np.concatenate((img[None], seg[None]))
    out_path = os.path.join(out_dir, '{}.npy'.format(six))
    np.save(out_path, out)

    with open(os.path.join(out_dir, 'meta_info_{}.pickle'.format(six)), 'wb') as handle:
        pickle.dump([out_path, class_id, str(six)], handle)


def generate_experiment(cf, exp_name, n_train_images, n_test_images, mode, class_diameters=(20, 20), noisy_bg=False):

    train_dir = os.path.join(cf.root_dir, exp_name, 'train')
    test_dir = os.path.join(cf.root_dir, exp_name, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # enforced distance between object center and image edge.
    foreground_margin = int(np.ceil(np.max(class_diameters) / 1.25))

    info = []
    info += [[train_dir, six, foreground_margin, class_diameters, mode, noisy_bg] for six in range(n_train_images)]
    info += [[test_dir, six, foreground_margin, class_diameters, mode, noisy_bg] for six in range(n_test_images)]

    print('starting creation of {} images'.format(len(info)))
    pool = Pool(processes=os.cpu_count()-1)
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
    stime = time.time()
    import sys
    sys.path.append("../..")
    import utils.exp_utils as utils

    parser = argparse.ArgumentParser()
    mode_choices = ['donuts_shape', 'donuts_pattern', 'circles_scale']
    parser.add_argument('-m', '--modes', nargs='+', type=str, default=mode_choices, choices=mode_choices)
    parser.add_argument('--noise', action='store_true', help="if given, add noise to the sample bg.")
    parser.add_argument('--n_train', type=int, default=1500, help="Nr. of train images to generate.")
    parser.add_argument('--n_test', type=int, default=1000, help="Nr. of test images to generate.")
    args = parser.parse_args()


    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    class_diameters = {
        'donuts_shape': (20, 20),
        'donuts_pattern': (20, 20),
        'circles_scale': (19, 20)
    }

    for mode in args.modes:
        generate_experiment(cf, mode + ("_noise" if args.noise else ""), n_train_images=args.n_train, n_test_images=args.n_test, mode=mode,
                            class_diameters=class_diameters[mode], noisy_bg=args.noise)


    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))



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

import numpy as np
from multiprocessing import Pool
import os
import subprocess


def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
    return case_identifiers


def convert_to_npy(npz_file):
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)['data']
        np.save(npz_file[:-3] + "npy", a)


def unpack_dataset(folder, threads=8):
    case_identifiers = get_case_identifiers(folder)
    p = Pool(threads)
    npz_files = [os.path.join(folder, i + ".npz") for i in case_identifiers]
    p.map(convert_to_npy, npz_files)
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [os.path.join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if os.path.isfile(i)]
    for n in npy_files:
        os.remove(n)


def mp_pack(inputs):
    ix , f = inputs
    file_path, source_dir, target_dir = f
    print('packing file number: {}'.format(ix))
    if 'npy' in file_path:
        source_path = os.path.join(source_dir, file_path)
        target_path = os.path.join(target_dir, file_path.split('.')[0] + '.npz')
        arr = np.load(source_path, mmap_mode='r')
        np.savez_compressed(target_path, data=arr)
        print('target_path', target_path)


if __name__ == '__main__':

    use_previous = False
    source_dir = '/mnt/hdd2/lidc/test_pp_rounding/'
    target_dir = '/mnt/hdd2/lidc/test_pp_rounding_packed/'

    if use_previous:
        file_list = [ii for ii in os.listdir(source_dir) if not ii in os.listdir(target_dir)]
    else:
        file_list = os.listdir(source_dir)
    info_list = [[ii, source_dir, target_dir] for ii in file_list]

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    pool = Pool(processes=12)
    p1 = pool.map(mp_pack, enumerate(info_list), chunksize=1)
    pool.close()
    pool.join()

    subprocess.call('cp {} {}'.format(os.path.join(source_dir, 'info_df.pickle'), os.path.join(target_dir, 'info_df.pickle')), shell=True)
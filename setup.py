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

from setuptools import find_packages, setup
import os, sys, subprocess


def parse_requirements(filename, exclude=[]):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#") and not line.split("==")[0] in exclude]

def pip_install(item):
    subprocess.check_call([sys.executable, "-m", "pip", "install", item])

def install_custom_ext(setup_path):
    try:
        pip_install(setup_path)
    except Exception as e:
        print("Could not install custom extension {} from source due to Error:\n{}\n".format(path, e) +
              "Trying to install from pre-compiled wheel.")
        dist_path = setup_path+"/dist"
        wheel_file = [fn for fn in os.listdir(dist_path) if fn.endswith(".whl")][0]
        pip_install(os.path.join(dist_path, wheel_file))

def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')




if __name__ == "__main__":

    req_file = "requirements.txt"
    custom_exts = ["nms-extension", "RoIAlign-extension-2D", "RoIAlign-extension-3D"]
    install_reqs = parse_requirements(req_file, exclude=custom_exts)

    setup(name='medicaldetectiontoolkit',
          version='0.1.0',
          url="https://github.com/MIC-DKFZ/medicaldetectiontoolkit",
          author='P. Jaeger, G. Ramien, MIC at DKFZ Heidelberg',
          license="Apache 2.0",
          description="Medical Object-Detection Toolkit.",
          classifiers=[
              "Development Status :: 4 - Beta",
              "Intended Audience :: Developers",
              "Programming Language :: Python :: 3.7"
          ],
          packages=find_packages(exclude=['test', 'test.*']),
          install_requires=install_reqs,
          python_requires=">=3.7"
          )

    custom_exts =  ["custom_extensions/nms", "custom_extensions/roi_align/2D", "custom_extensions/roi_align/3D"]
    for path in custom_exts:
        try:
            install_custom_ext(path)
        except Exception as e:
            print("FAILED to install custom extension {} due to Error:\n{}".format(path, e))

    clean()
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
import os

def parse_requirements(filename, exclude=[]):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#") and not line.split("==")[0] in exclude]

def install_custom_ext(setup_path):
    os.system("python "+setup_path+" install")
    return

def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

req_file = "requirements.txt"
custom_exts = ["nms-extension", "RoIAlign-extension-2D", "RoIAlign-extension-3D"]
install_reqs = parse_requirements(req_file, exclude=custom_exts)



setup(name='medicaldetectiontoolkit',
      version='0.0.1',
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
      )

custom_exts =  ["custom_extensions/nms", "custom_extensions/roi_align"]
for path in custom_exts:
    setup_path = os.path.join(path, "setup.py")
    try:
        install_custom_ext(setup_path)
    except Exception as e:
        print("FAILED to install custom extension {} due to Error:\n{}".format(path, e))

clean()
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
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

import sys
from glob import glob
import shutil
import setuptools
import tensorflow as tf

TF_INCLUDE, TF_CFLAG = tf.sysconfig.get_compile_flags()
TF_INCLUDE = TF_INCLUDE.split("-I")[1]

TF_LIB_INC, TF_SO_LIB = tf.sysconfig.get_link_flags()
TF_SO_LIB = TF_SO_LIB.replace(
    "-l:libtensorflow_framework.1.dylib", "-ltensorflow_framework.1"
)
TF_LIB_INC = TF_LIB_INC.split("-L")[1]
TF_SO_LIB = TF_SO_LIB.split("-l")[1]

complie_args = [TF_CFLAG, "-fPIC", "-shared", "-O2", "-std=c++11"]
if sys.platform == "darwin":  # Mac os X before Mavericks (10.9)
    complie_args.append("-stdlib=libc++")

module = setuptools.Extension(
    "athena.transform.feats.ops.x_ops",
    sources=glob("athena/transform/feats/ops/kernels/*.cc"),
    depends=glob("athena/transform/feats/ops/kernels/*.h"),
    extra_compile_args=complie_args,
    include_dirs=["athena/transform/feats/ops", TF_INCLUDE],
    library_dirs=[TF_LIB_INC],
    libraries=[TF_SO_LIB],
    language="c++",
)


with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="athena",
        version="0.1.0",
        author="ATHENA AUTHORS",
        author_email="athena@gmail.com",
        description="for speech recognition",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/didichuxing/athena",
        packages=setuptools.find_packages(),
        package_data={"": ["x_ops*.so", "*.vocab", "sph2pipe"]},
        exclude_package_data={"feats": ["*_test*"]},
        ext_modules=[module],
        python_requires=">=3",
    )
path = glob("build/lib.*/athena/transform/feats/ops/x_ops.*.so")
shutil.copy(path[0], "athena/transform/feats/ops/x_ops.so")

#!/bin/bash

#Usage:
# -->not true?: this script has to be started from the same directory the python files called below lie in (e.g. exec.py lies in meddetectiontkit).
# part of the slurm-job name you pass to sbatch will be the experiment folder's name.
# you need to pass 3 positional arguments to this script (cluster_runner_..sh #1 #2 #3):
# -#1 source directory in which main source code (framework) is located (e.g. medicaldetectiontoolkit/)
# -#2 the exp_dir where job-specific code was copied before by create_exp and exp results are safed by exec.py
# -#3 absolute path to dataset-specific code in source dir
# -#4 mode to run
# -#5 folds to run on

source_dir=${1}
exp_dir=${2}
dataset_abs_path=${3}
mode=${4}
folds=${5}
resume=$6

job_dir=/ssd/ramien/${LSB_JOBID}

tmp_dir_data=${job_dir}/data
mkdir $tmp_dir_data

tmp_dir_cache=${job_dir}/cache
mkdir $tmp_dir_cache
CUDA_CACHE_PATH=$tmp_dir_cache
export CUDA_CACHE_PATH


#data must not lie permantly on nodes' ssd, only during training time
#needs to be named with the SLURM_JOB_ID to not be automatically removed
#can permanently lie on /datasets drive --> copy from there before every experiment
#files on datasets are saved as npz (compressed) --> use data_manager.py to copy and unpack into .npy; is done implicitly in exec.py

#(tensorboard --logdir ${exp_dir}/.. --port 1337 || echo "tboard startup failed")& # || tensorboard --logdir ${exp_dir}/.. --port 1338)&
#tboard_pid=$!


export OMP_NUM_THREADS=1 # this is a work-around fix for batchgenerators to deal with numpy-inherent multi-threading.

launch_opts="${source_dir}/exec.py --use_stored_settings --server_env --exp_source ${dataset_abs_path} --data_dest ${tmp_dir_data} --exp_dir ${exp_dir} --mode ${mode}"

if [ ! -z "${resume}" ]; then
  launch_opts="${launch_opts} --resume"
  echo "Resuming from checkpoint(s)."
fi

if [ ! -z "${folds}" ]; then
  launch_opts="${launch_opts} --folds ${folds}"
fi

echo "submitting with ${launch_opts}"
python ${launch_opts}






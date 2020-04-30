#!/bin/bash
#wrapper for cluster_runner_....sh which copies job-specific, frequently changing files (e.g. configs.py) before the actual sbatch job 
#is submitted since the job might pend in queue before execution --> hazard of job-specific files being unintentionally changed during queue wait time. 
#positonal
# -arg #1 identifies the folder name of the dataset-related code (e.g. >toy_exp< or >lidc_exp<) within the code source directory
# -arg #2 is the experiment and first part of the job name,
# optional args and flags:
# -c / --create: (flag) whether to create the exp, i.e., if this is a new start of the exp with configs etc from source dir.
# -f / --folds FOLDS: (option) fold(s) to run on (FOLDS needs to be only one int or string of multiple ints separated by space), default None (-->set to all in config)
# -m / --mode MODE: (option) string, one of "train", "train_test", "test", defaults to "train_test"
# -p / --exp_parent_dir: (option) name of parent_dir rel to dataset folder on cluster. exp_dir is exp_parent_dir/exp_name, if not given defaults to "experiments"
# -q / --queue: (option) which queue (-q parameter for bsub) to send job to. default: gputest. others: gputest-short (max 5h jobs). 
# -w / --which: (option) same as argument -m to bsub; host or host list (string separated by space) to send the job to.
# 		use nodenameXX where XX==nr of node or nodenameXX,nodenameYY,... or nodename[XX-YY]. nodename is e.g. e132-comp.
# --gmem: (option) how much gpu memory to request for job (in gigabytes), defaults to 11.9. Currently, the smaller nodes have 11.9G, the larger ones 31.7G.
# --resume: (flag) only with explicit fold argument, if set, resumes from checkpoint in exp_dir/fold_x/last_state.pth.
# --no_parallel: (flag) if set, folds won't start as parallel jobs on cluster, but run sequentially in one job.

dataset_name="${1}"
exp_name="${2}"

#arguments not passed, e.g. $7 if no seventh argument, are null.
if [ ! -z "${18}" ]; then #-z checks if is null string
 echo "Error: Received too many arguments."
 exit
fi

#make args optional: move up if some args are missing inbetween
while [ ${#} -gt 2 ]; do
  case "${3}" in
		-c|--create)
      		create_exp="c"
			shift
      		;;
		-f|--folds)
			folds="${4}"
			shift; shift
			;;
		-m|--mode)
			mode="${4}"
			shift; shift
			;;
		-p|--exp_parent_dir)
			exp_parent_dir="${4}"
			shift; shift			
			;;
		-q|--queue)
			queue="${4}"
			shift; shift			
			;;
		-w|--which)
			which="${4}"
			shift; shift			
			;;
		-R|--resource)
			resource="${4}"
			shift; shift			
			;;
		--gmem)
			gmem="${4}"
			shift; shift
			;;
		--resume)
			resume=true
			shift
			;;
		--no_parallel)
			no_parallel=true
			shift
			;;
    *)
			echo "Invalid argument/option passed: ${3}"
			exit 1
			;;
  esac
done

# default values
if [ -z ${exp_parent_dir} ]; then 
	exp_parent_dir="experiments"
fi

if [ -z ${mode} ]; then 
	mode="train_test"
fi

if [ -z ${queue} ]; then 
	queue="gputest"
fi


if [ -z ${gmem} ]; then 
	gmem="11"
fi


root_dir=/home/ramien #assumes /home/ramien exists
#medicaldetectiontoolkit
source_dir=${root_dir}/mdt-public

dataset_abs_path=${source_dir}/experiments/${dataset_name} #set as second argument passed to this script
exp_parent_dir=/datasets/datasets_ramien/${dataset_name}/${exp_parent_dir}
exp_dir=${exp_parent_dir}/${exp_name}

#activate virtualenv that has all the packages:
source_dl="module load python/3.7.0; module load gcc/7.2.0; source ${root_dir}/.virtualenvs/mdt/bin/activate;"

eval ${source_dl}

# directly from prep node:
create_cmd="python ${source_dir}/exec.py --server_env --mode create_exp --exp_dir ${exp_dir} --exp_source ${dataset_abs_path};"


#if create_exp, check if would overwrite existing exp_dir
if [ ! -z ${create_exp} ] && [ ${create_exp} = "c" ]; then #-n doesnt work as replacement for !-z
	if [ -d ${exp_dir} ]; then
		echo "Please confirm to overwrite exp ${exp_name} settings, (Y/n): "; read confirmation
		if ([ "${confirmation}" = "y" ] || [ "${confirmation}" = "yes" ] || [ "${confirmation}" = "Y" ] || [ -z "${confirmation}" ]); then
				echo "Overwriting ${exp_name}"
		else
				echo "Exiting due to overwrite denial. Adjust options."
				exit
		fi
	fi
	#echo "opts: name ${exp_name}, ${source_dir}/exec.py --server_env --mode create_exp --exp_dir ${exp_dir} --exp_source ${dataset_abs_path}"
	echo "Creating ${exp_name}"
	eval ${create_cmd}
else
	if [ ! -d ${exp_dir} ]; then
		echo "Experiment directory ${exp_dir} does not exist."
		echo "Run create_exp? (Y/n): "; read confirmation
			if ([ "${confirmation}" = "y" ] || [ "${confirmation}" = "yes" ] || [ "${confirmation}" = "Y" ] || [ -z "${confirmation}" ]); then
				echo "Creating ${exp_name}"
				eval ${create_cmd}
			fi
	fi
fi

#if not create_exp, check if would overwrite existing folds (possibly valuable trained params!)
if [ -z ${create_exp} ] && ([ ${mode} = "train" ] || [ ${mode} = "train_test" ]) && [ -z "${resume}" ]; then
	for f in ${folds}; do #if folds is null this check won't apply and folds will be quietly overwritten.
		if [ -d ${exp_dir}/fold_${f} ]; then #-d checks if is dir
			echo "please confirm to overwrite fold_${f}, (Y/n):"; read confirmation
			if ([ "${confirmation}" = "y" ] || [ "${confirmation}" = "yes" ] || [ "${confirmation}" = "Y" ] || [ -z "${confirmation}" ]); then
				echo "Overwriting "${exp_name}/fold_${f}
			else
				echo "Exiting due to overwrite denial. Adjust options."
				exit
			fi
		fi
	done
fi



bsub_opts="bsub -N -q ${queue} -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=${gmem}G"
if [ ! -z "$resource" ]; then
	bsub_opts=$bsub_opts $resource
fi
if [ ! -z ${which} ]; then
	bsub_opts="${bsub_opts} -m ${which}"
fi

#----- parallel/separate fold jobs (each fold in a single job) -----------
if [ ! -z "${folds}" ] && [ -z ${no_parallel} ]; then #WHY do i need to convert to string again?
	for f in ${folds}; do
		out_file=${exp_dir}/logs/fold_${f}_lsf_output.out
		bsub_opts="$bsub_opts -J '${dataset_name} ${exp_name}  fold ${f} ${mode}' -oo '${out_file}'"
		eval "${bsub_opts} sh cluster_runner_meddec.sh ${source_dir} ${exp_dir} ${dataset_abs_path} ${mode} ${f} ${resume}"
	done

#----- consecutive folds job (all folds in one single job) -----------
else 
	if [ ! -z ${resume} ]; then
		echo "You need to explicitly specify folds if you would like to resume from a checkpoint. Exiting."
		exit
	fi
	out_file=${exp_dir}/logs/lsf_output.out
	bsub_opts="$bsub_opts -J '${dataset_name} ${exp_name}  folds ${folds} ${mode}' -oo '${out_file}'"
	eval "${bsub_opts} sh cluster_runner_meddec.sh ${source_dir} ${exp_dir} ${dataset_abs_path} ${mode} ${folds} ${resume}"
	echo "Started in no parallel, folds:" ${folds}
fi




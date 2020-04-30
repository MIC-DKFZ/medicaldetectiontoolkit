#!/bin/bash

# wrapper around lsf-scheduler bpeek command writing bpeek to file and opening it.
# amend default output dir and editor as desired.
# positonal
#  -arg #1 lsf job id

job_id="${1}"
out_dir=$HOME/lsf_peek_output
out_file="${out_dir}/${job_id}.out"


mkdir -p ${out_dir}

eval "bpeek ${job_id} > ${out_file}"

nano ${out_file}
#tail -F ${out_file}


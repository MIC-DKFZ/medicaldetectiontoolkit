
#cluster sync
rootp=/home/gregor/Documents
server=ramien@odcf-lsf01.dkfz.de
#server=ramien@e132-comp04
serverroot=${server}:/home/ramien

#---- medicaldetectiontoolkit -----
codep=${rootp}/mdt-public
server_codep=${serverroot}/mdt-public

rsync -avhe "ssh -i /home/gregor/.ssh/id_rsa" ${rootp}/mdt-public/shell_scripts/cluster_runner_meddec.sh ${rootp}/mdt-public/shell_scripts/job_starter.sh ${server_codep}
rsync -avhe "ssh -i /home/gregor/.ssh/id_rsa" ${rootp}/environmental/job_scheduler,cluster/bpeek_wrapper.sh ${serverroot}

# add/remove --include 'custom_extension/**/*.whl' for compiled c++/CUDA exts

rsync -avhe "ssh -i /home/gregor/.ssh/id_rsa" --include '*/' --include '*.py' --include '*.cpp' --include '*.cu' --include '*.h' --include 'requirements.txt' --exclude '*' --prune-empty-dirs ${codep} ${serverroot}


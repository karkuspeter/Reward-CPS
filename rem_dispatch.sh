#!/bin/sh
#
REMDIR="~/GIT-hpc$1/Reward-CPS"
REMOTE="e0001940@atlas6-c01.nus.edu.sg"
FILE=$2
COMMIT=$3
if [ $# -lt 2 ]
then
FILE='batch_run.m'
fi
if [ $# -lt 3 ]
then
COMMIT='master'
fi

ssh ${REMOTE} "( cd ${REMDIR} ; git reset --hard origin/${COMMIT}; git fetch; git checkout ${COMMIT}; git reset --hard origin/${COMMIT}; git pull; git reset --hard)"
scp ${FILE} ${REMOTE}:${REMDIR}/batch_run.m

while true; do
    read -p "Continue with dispatch?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

ssh ${REMOTE} "rm ${REMDIR}/stdout.o ${REMDIR}/running.o ${REMDIR}/results/*"
ssh ${REMOTE} "( cd ${REMDIR} ;"'bsub -q matlab -o stdout.o "matlab -nosplash -nodisplay -r batch_run"'"; echo ' ' > running.o )"

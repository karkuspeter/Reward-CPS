#!/bin/sh
#
REMDIR="~/GIT-hpc$1/Reward-CPS"
REMOTE="e0001940@atlas6-c01.nus.edu.sg"

NOW=$(date +"%m-%d-%H-%M-%S")
LOCDIR="../../results/${NOW}"
mkdir ${LOCDIR}

scp ${REMOTE}:${REMDIR}/batch_run.m ${LOCDIR}/
scp ${REMOTE}:${REMDIR}/stdout.o ${LOCDIR}/
scp ${REMOTE}:${REMDIR}/results/hyper-* ${LOCDIR}/
ssh ${REMOTE} "( cd ${REMDIR} ; git log )" >> ${LOCDIR}/gitlog.o

rm ../../results/latest/*
cp ${LOCDIR}/* ../../results/latest/

echo "Saved to ${LOCDIR}"

while true; do
    read -p "Continue with cleanup?" yn
    case $yn in
        [Yy]* ) ssh ${REMOTE} "rm ${REMDIR}/stdout.o ${REMDIR}/running.o ${REMDIR}/results/*" ; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
#!/bin/sh
#
REMOTE="e0001940@atlas6-c01.nus.edu.sg"

ssh ${REMOTE} "bjobs"

for i in `seq 1 10`;
do
REMDIR="~/GIT-hpc$i/Reward-CPS"

if ! ssh $REMOTE stat $REMDIR \> /dev/null 2\>\&1
            then
                    continue
fi

if ssh $REMOTE stat $REMDIR/stdout.o \> /dev/null 2\>\&1
then
                    echo "$i finished";
else
	if ssh $REMOTE stat $REMDIR/running.o \> /dev/null 2\>\&1
	then
		echo "$i running"
	else
		echo "$i free"
	fi
fi
ssh ${REMOTE} "ls $REMDIR/results/"


done  
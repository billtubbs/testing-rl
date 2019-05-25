#!/bin/sh

# Shell script to launch RL model testing experiments
# This is a template to use as a starting point for creating
# your own script files. Please do not overwrite it.

# Instructions for use:
# Make sure the paths are correct and execute from the command
# line using:
# $ ./yourscript.sh
# You will have to change file permissions before you can
# execute it:
# $ chmod +x yourscript.sh
# To automate the execution of multiple scipts use the
# jobdispatcher.py tool.

# Launch the Python executables from parent directory
cp main.py ../main.py
cp parsers.py ../parsers.py

MODEL='DDPG'
ENVIRONMENT='LunarLanderContinuous-v2'

# Setup
TIMESTAMP=`date +%y%m%d%H%M%S`  # Use this in LOGDIR (optional)
FILENAME=`basename "$0"`  # Use this in --label (optional)
BASELOG='../logs/'$ENVIRONMENT/$MODEL  # Use the env and model names
LOGDIR=$BASELOG/$TIMESTAMP
#SCRATCH='/data/scratch/'$TIMESTAMP

mkdir -p $DATADIR
#mkdir -p $SCRATCH
mkdir -p $BASELOG
mkdir -p $LOGDIR

#ln -s $SCRATCH $LOGDIR

python -u ../main.py \
    --env $ENVIRONMENT \
    --model $MODEL \
    --name $FILENAME \
    --comment 'Template script' \
    --n_steps 5000 \
    --seed 1 \
    --reset-timesteps \
    --log-interval 10 \
    --log_dir $LOGDIR \
    --save-model \
#    --overwrite \
#    --plots \
#    --model-args {} \
    > $LOGDIR/log.out 2>&1 # Write stdout directly to log.out
                           # if you want to see results in real time,
                           # use tail -f

#rm $LOGDIR
#mv $SCRATCH $LOGDIR
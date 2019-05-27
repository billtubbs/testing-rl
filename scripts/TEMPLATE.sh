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

# The Python executables will be launched from the parent
# directory of job-dispatcher.  To be safe, use absolute path
cp ~/code/openai/testing-rl/main.py ~/code/main.py
cp ~/code/openai/testing-rl/parsers.py ~/code/parsers.py

MODEL='PPO2'
ENVIRONMENT='CartPole-BT-dL-v0'

# Setup
TIMESTAMP=`date +%y%m%d%H%M%S`  # Use this in LOGDIR (optional)
FILENAME=`basename "$0"`  # Use this in --name (optional)
BASELOG='../../experiments/logs/'$ENVIRONMENT/$MODEL  # Use the env and model names
LOGDIR=$BASELOG/$TIMESTAMP
#SCRATCH='../../experiments/logs/scratch/'$TIMESTAMP

#mkdir -p $SCRATCH
mkdir -p $BASELOG
mkdir -p $LOGDIR

# Save a copy of this script
cp `basename "$0"` $LOGDIR/`basename "$0"`

#ln -s $SCRATCH $LOGDIR

python -u ../main.py \
    --env $ENVIRONMENT \
    --model $MODEL \
    --name $FILENAME \
    --comment 'Early test runs' \
    --n_steps 100000 \
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

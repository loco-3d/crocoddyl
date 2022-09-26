#!/bin/bash

LOGPATH="$( cd "$(dirname "$0")" ; pwd -P )"
EXAMPLEPATH=${LOGPATH}/..

# If PYTHON_EXECUTABLE has not been set, then try to determine using `which`
if [ -z $PYTHON_EXECUTABLE ] ; then
  tmp=$(which python)
  if [ $? -eq 0 ] ; then
    echo "Using $tmp"
    PYTHON_EXECUTABLE=$tmp
  else
    tmp=$(which python3)
    if [ $? -eq 0 ] ; then
      echo "Using $tmp"
      PYTHON_EXECUTABLE=$tmp
    else
      echo "Could not determine PYTHON_EXECUTABLE!"
    fi
  fi
else
  echo "PYTHON_EXECUTABLE set, using $PYTHON_EXECUTABLE"
fi

echo "Updating the log files ..."
update_logfile() {
  FILENAME=$1
  echo "    ${FILENAME}"
  ${PYTHON_EXECUTABLE} -u ${EXAMPLEPATH}/${FILENAME}.py > ${LOGPATH}/${FILENAME}.log
}

update_logfile "arm_manipulation"
update_logfile "arm_manipulation_id"
update_logfile "bipedal_walk"
update_logfile "bipedal_walk_id"
update_logfile "bipedal_walk_ubound"
update_logfile "boxfddp_vs_boxddp"
update_logfile "double_pendulum"
update_logfile "double_pendulum_id"
update_logfile "humanoid_manipulation"
update_logfile "humanoid_manipulation_ubound"
update_logfile "humanoid_taichi"
update_logfile "quadrotor"
update_logfile "quadrotor_id"
update_logfile "quadrotor_ubound"
update_logfile "quadrupedal_gaits"
update_logfile "quadrupedal_gaits_id"
update_logfile "quadrupedal_walk_ubound"

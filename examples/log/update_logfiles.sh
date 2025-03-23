#!/bin/bash

# Run Python code to get the version of pinocchio
VERSION=$(python -c "import pinocchio; print(pinocchio.__version__)" 2>/dev/null)
# Check if the version matches
EXPECTED_VERSION="3.2.0"
if [ "$VERSION" != "$EXPECTED_VERSION" ]; then
    # Print warning message in yellow
    echo -e "\033[33mWarning: Pinocchio version is $VERSION, but it is expected $EXPECTED_VERSION.\033[0m"
else
    echo "pinocchio version: $VERSION"
fi

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

update_logfile "arm_manipulation_fwddyn"
update_logfile "arm_manipulation_invdyn"
update_logfile "bipedal_gaits_fwddyn"
update_logfile "bipedal_gaits_invdyn"
update_logfile "bipedal_walk_ubound"
update_logfile "boxfddp_vs_boxddp"
update_logfile "double_pendulum_fwddyn"
update_logfile "double_pendulum_invdyn"
update_logfile "humanoid_manipulation"
update_logfile "humanoid_manipulation_ubound"
update_logfile "humanoid_taichi"
update_logfile "quadrotor_fwddyn"
update_logfile "quadrotor_invdyn"
update_logfile "quadrotor_ubound"
update_logfile "quadrupedal_gaits_fwddyn"
update_logfile "quadrupedal_gaits_invdyn"
update_logfile "quadrupedal_walk_ubound"

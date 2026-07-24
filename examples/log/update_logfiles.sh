#!/bin/bash

# Run Python code to get the version of pinocchio
VERSION=$(python -c "import pinocchio; print(pinocchio.__version__)" 2>/dev/null)
# Check if the version matches
EXPECTED_VERSION="4.1.0"
if [ "$VERSION" != "$EXPECTED_VERSION" ]; then
    # Print warning message in yellow
    echo -e "\033[33mWarning: Pinocchio version is $VERSION, but it is expected $EXPECTED_VERSION.\033[0m"
else
    echo "pinocchio version: $VERSION"
fi

LOGPATH="$( cd "$(dirname "$0")" ; pwd -P )"
EXAMPLEPATH=${LOGPATH}/..
TIMEOPTEXAMPLEPATH=${EXAMPLEPATH}/timeopt
TIMEOPTLOGPATH=${TIMEOPTEXAMPLEPATH}/log
ESTIMATIONEXAMPLEPATH=${EXAMPLEPATH}/estimation
ESTIMATIONLOGPATH=${ESTIMATIONEXAMPLEPATH}/log

mkdir -p ${TIMEOPTLOGPATH} ${ESTIMATIONLOGPATH}

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

update_timeopt_logfile() {
  FILENAME=$1
  echo "    timeopt/${FILENAME}"
  ${PYTHON_EXECUTABLE} -u ${TIMEOPTEXAMPLEPATH}/${FILENAME}.py > ${TIMEOPTLOGPATH}/${FILENAME}.log
}

update_estimation_logfile() {
  FILENAME=$1
  echo "    estimation/${FILENAME}"
  ${PYTHON_EXECUTABLE} -u ${ESTIMATIONEXAMPLEPATH}/${FILENAME}.py > ${ESTIMATIONLOGPATH}/${FILENAME}.log
}

update_logfile "aerial_manipulator_fwddyn"
update_logfile "aerial_manipulator_invdyn"
update_logfile "biped_gaits_fwddyn"
update_logfile "biped_gaits_invdyn"
update_logfile "biped_pose_fwddyn"
update_logfile "biped_pose_invdyn"
update_logfile "biped_walk_ubound"
update_logfile "boxfddp_vs_boxddp"
update_logfile "double_pendulum_fwddyn"
update_logfile "double_pendulum_invdyn"
update_logfile "double_pendulum_continuous_fwddyn"
update_logfile "double_pendulum_continuous_invdyn"
update_logfile "humanoid_backflip_fwddyn"
update_logfile "humanoid_backflip_invdyn"
update_logfile "humanoid_bar_fwddyn"
update_logfile "humanoid_bar_invdyn"
update_logfile "humanoid_frontflip_fwddyn"
update_logfile "humanoid_frontflip_invdyn"
update_logfile "humanoid_handstanding_fwddyn"
update_logfile "humanoid_handstanding_invdyn"
update_logfile "humanoid_manipulation_ubound"
update_logfile "humanoid_manipulation"
update_logfile "humanoid_taichi"
update_logfile "humanoid_walking_on_hands_fwddyn"
update_logfile "humanoid_walking_on_hands_invdyn"
update_logfile "lqr_fwddyn"
update_logfile "lqr_invdyn"
update_logfile "manipulator_fwddyn"
update_logfile "manipulator_invdyn"
update_logfile "marine_manipulator_fwddyn"
update_logfile "marine_manipulator_invdyn"
update_logfile "quadrotor_fwddyn"
update_logfile "quadrotor_invdyn"
update_logfile "quadrotor_ubound"
update_logfile "quadruped_gaits_fwddyn"
update_logfile "quadruped_gaits_invdyn"
update_logfile "quadruped_pose_fwddyn"
update_logfile "quadruped_pose_invdyn"
update_logfile "quadruped_walk_ubound"

update_timeopt_logfile "double_pendulum_timeopt_fwddyn"
update_timeopt_logfile "double_pendulum_timeopt_invdyn"
update_timeopt_logfile "manipulator_timeopt_fwddyn"
update_timeopt_logfile "manipulator_timeopt_invdyn"
update_timeopt_logfile "quadrotor_timeopt_fwddyn"
update_timeopt_logfile "quadrotor_timeopt_invdyn"
update_timeopt_logfile "quadruped_gait_timeopt_fwddyn"
update_timeopt_logfile "quadruped_gait_timeopt_invdyn"
update_timeopt_logfile "quadruped_pose_timeopt_fwddyn"
update_timeopt_logfile "quadruped_pose_timeopt_invdyn"

update_estimation_logfile "double_pendulum_inertialparam_fwddyn"
update_estimation_logfile "double_pendulum_inertialparam_invdyn"
update_estimation_logfile "manipulator_inertialparam_fwddyn"
update_estimation_logfile "manipulator_inertialparam_invdyn"
update_estimation_logfile "quadrotor_inertialparam_fwddyn"
update_estimation_logfile "quadrotor_inertialparam_invdyn"
update_estimation_logfile "biped_gaits_friction_inertialparam_fwddyn"
update_estimation_logfile "biped_gaits_friction_inertialparam_invdyn"
update_estimation_logfile "quadruped_gaits_friction_inertialparam_fwddyn"
update_estimation_logfile "quadruped_gaits_friction_inertialparam_invdyn"

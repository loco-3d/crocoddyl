
LOGPATH="$( cd "$(dirname "$0")" ; pwd -P )"
EXAMPLEPATH=${LOGPATH}/..

echo "Updating the log files ..."
update_logfile() {
  FILENAME=$1
  echo "    ${FILENAME}"
  python -u ${EXAMPLEPATH}/${FILENAME}.py > ${LOGPATH}/${FILENAME}.log
}

update_logfile "arm_manipulation"
update_logfile "bipedal_walk"
update_logfile "bipedal_walk_ubound"
update_logfile "boxfddp_vs_boxddp"
update_logfile "double_pendulum"
update_logfile "humanoid_manipulation"
update_logfile "humanoid_manipulation_ubound"
update_logfile "humanoid_taichi"
update_logfile "quadrotor"
update_logfile "quadrotor_ubound"
update_logfile "quadrupedal_gaits"
update_logfile "quadrupedal_walk_ubound"

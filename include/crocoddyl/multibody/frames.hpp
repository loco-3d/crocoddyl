
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRAMES_HPP_
#define CROCODDYL_MULTIBODY_FRAMES_HPP_

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>

namespace crocoddyl {

struct FrameTranslation {
  FrameTranslation(unsigned int const& frame, const Eigen::Vector3d& oxf) : frame(frame), oxf(oxf) {}
  FrameTranslation(long unsigned int const& frame, const Eigen::Vector3d& oxf)
      : frame(static_cast<unsigned int>(frame)), oxf(oxf) {}
  unsigned int frame;
  Eigen::Vector3d oxf;
};

struct FrameRotation {
  FrameRotation(unsigned int const& frame, const Eigen::Matrix3d& oRf) : frame(frame), oRf(oRf) {}
  FrameRotation(long unsigned int const& frame, const Eigen::Matrix3d& oRf)
      : frame(static_cast<unsigned int>(frame)), oRf(oRf) {}
  unsigned int frame;
  Eigen::Matrix3d oRf;
};

struct FramePlacement {
  FramePlacement(unsigned int const& frame, const pinocchio::SE3& oMf) : frame(frame), oMf(oMf) {}
  FramePlacement(long unsigned int const& frame, const pinocchio::SE3& oMf)
      : frame(static_cast<unsigned int>(frame)), oMf(oMf) {}
  unsigned int frame;
  pinocchio::SE3 oMf;
};

struct FrameMotion {
  FrameMotion(unsigned int const& frame, const pinocchio::Motion& oMf) : frame(frame), oMf(oMf) {}
  FrameMotion(long unsigned int const& frame, const pinocchio::Motion& oMf)
      : frame(static_cast<unsigned int>(frame)), oMf(oMf) {}
  unsigned int frame;
  pinocchio::Motion oMf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

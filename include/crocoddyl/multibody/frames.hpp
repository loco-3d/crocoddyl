
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

namespace crocoddyl {

struct FrameTranslation {
  FrameTranslation(const unsigned int& frame, const Eigen::Vector3d& oxf) : frame(frame), oxf(oxf) {}
  unsigned int frame;
  Eigen::Vector3d oxf;
};

struct FramePlacement {
  FramePlacement(const unsigned int& frame, const pinocchio::SE3& oMf) : frame(frame), oMf(oMf) {}
  unsigned int frame;
  pinocchio::SE3 oMf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

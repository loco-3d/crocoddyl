
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
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

typedef std::size_t FrameIndex;

struct FrameTranslation {
  FrameTranslation(const FrameIndex& frame, const Eigen::Vector3d& oxf) : frame(frame), oxf(oxf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameTranslation& X) {
    os << "      frame: " << X.frame << std::endl << "translation: " << std::endl << X.oxf.transpose() << std::endl;
    return os;
  }

  FrameIndex frame;
  Eigen::Vector3d oxf;
};

struct FrameRotation {
  FrameRotation(const FrameIndex& frame, const Eigen::Matrix3d& oRf) : frame(frame), oRf(oRf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameRotation& X) {
    os << "   frame: " << X.frame << std::endl << "rotation: " << std::endl << X.oRf << std::endl;
    return os;
  }

  FrameIndex frame;
  Eigen::Matrix3d oRf;
};

struct FramePlacement {
  FramePlacement(const FrameIndex& frame, const pinocchio::SE3& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FramePlacement& X) {
    os << "    frame: " << X.frame << std::endl << "placement: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::SE3 oMf;
};

struct FrameMotion {
  FrameMotion(const FrameIndex& frame, const pinocchio::Motion& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameMotion& X) {
    os << " frame: " << X.frame << std::endl << "motion: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::Motion oMf;
};

struct FrameForce {
  FrameForce(const FrameIndex& frame, const pinocchio::Force& oFf) : frame(frame), oFf(oFf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameForce& X) {
    os << "frame: " << X.frame << std::endl << "force: " << std::endl << X.oFf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::Force oFf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

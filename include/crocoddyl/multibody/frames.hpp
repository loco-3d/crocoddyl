
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRAMES_HPP_
#define CROCODDYL_MULTIBODY_FRAMES_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

typedef std::size_t FrameIndex;

template <typename _Scalar>
struct FrameTranslationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Vector3s Vector3s;

  FrameTranslationTpl(const FrameIndex& frame, const Vector3s& oxf) : frame(frame), oxf(oxf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameTranslationTpl<Scalar>& X) {
    os << "      frame: " << X.frame << std::endl << "translation: " << std::endl << X.oxf.transpose() << std::endl;
    return os;
  }

  FrameIndex frame;
  Vector3s oxf;
};

template <typename _Scalar>
struct FrameRotationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Matrix3s Matrix3s;

  FrameRotationTpl(const FrameIndex& frame, const Matrix3s& oRf) : frame(frame), oRf(oRf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameRotationTpl<Scalar>& X) {
    os << "   frame: " << X.frame << std::endl << "rotation: " << std::endl << X.oRf << std::endl;
    return os;
  }

  FrameIndex frame;
  Matrix3s oRf;
};

template <typename _Scalar>
struct FramePlacementTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;

  FramePlacementTpl(const FrameIndex& frame, const pinocchio::SE3Tpl<Scalar>& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FramePlacementTpl<Scalar>& X) {
    os << "    frame: " << X.frame << std::endl << "placement: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::SE3Tpl<Scalar> oMf;
};

template <typename _Scalar>
struct FrameMotionTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;

  FrameMotionTpl(const FrameIndex& frame, const pinocchio::MotionTpl<Scalar>& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameMotionTpl<Scalar>& X) {
    os << " frame: " << X.frame << std::endl << "motion: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::MotionTpl<Scalar> oMf;
};

template <typename _Scalar>
struct FrameForceTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;

  FrameForceTpl(const FrameIndex& frame, const pinocchio::ForceTpl<Scalar>& oFf) : frame(frame), oFf(oFf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameForceTpl<Scalar>& X) {
    os << "frame: " << X.frame << std::endl << "force: " << std::endl << X.oFf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::ForceTpl<Scalar> oFf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

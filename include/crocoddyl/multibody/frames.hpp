
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
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/core/mathbase.hpp"

#include <pinocchio/multibody/fwd.hpp>
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

  explicit FrameTranslationTpl() : id(0), translation(Vector3s::Zero()) {}
  FrameTranslationTpl(const FrameTranslationTpl<Scalar>& value) : id(value.id), translation(value.translation) {}
  FrameTranslationTpl(const FrameIndex& id, const Vector3s& translation) : id(id), translation(translation) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameTranslationTpl<Scalar>& X) {
    os << "         id: " << X.id << std::endl
       << "translation: " << std::endl
       << X.translation.transpose() << std::endl;
    return os;
  }

  template <typename OtherScalar>
  bool operator==(const FrameTranslationTpl<OtherScalar>& other) const {
    return id == other.id && translation == other.translation;
  }

  FrameIndex id;
  Vector3s translation;
};

template <typename _Scalar>
struct FrameRotationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Matrix3s Matrix3s;

  explicit FrameRotationTpl() : id(0), rotation(Matrix3s::Identity()) {}
  FrameRotationTpl(const FrameRotationTpl<Scalar>& value) : id(value.id), rotation(value.rotation) {}
  FrameRotationTpl(const FrameIndex& id, const Matrix3s& rotation) : id(id), rotation(rotation) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameRotationTpl<Scalar>& X) {
    os << "      id: " << X.id << std::endl << "rotation: " << std::endl << X.rotation << std::endl;
    return os;
  }

  template <typename OtherScalar>
  bool operator==(const FrameRotationTpl<OtherScalar>& other) const {
    return id == other.id && rotation == other.rotation;
  }

  FrameIndex id;
  Matrix3s rotation;
};

template <typename _Scalar>
struct FramePlacementTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  explicit FramePlacementTpl() : id(0), placement(SE3::Identity()) {}
  FramePlacementTpl(const FramePlacementTpl<Scalar>& value) : id(value.id), placement(value.placement) {}
  FramePlacementTpl(const FrameIndex& id, const SE3& placement) : id(id), placement(placement) {}

  template <typename OtherScalar>
  bool operator==(const FramePlacementTpl<OtherScalar>& other) const {
    return id == other.id && placement == other.placement;
  }

  friend std::ostream& operator<<(std::ostream& os, const FramePlacementTpl<Scalar>& X) {
    os << "       id: " << X.id << std::endl << "placement: " << std::endl << X.placement << std::endl;
    return os;
  }

  FrameIndex id;
  pinocchio::SE3Tpl<Scalar> placement;
};

template <typename _Scalar>
struct FrameMotionTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  explicit FrameMotionTpl() : id(0), motion(Motion::Zero()), reference(pinocchio::LOCAL) {}
  FrameMotionTpl(const FrameMotionTpl<Scalar>& value)
      : id(value.id), motion(value.motion), reference(value.reference) {}
  FrameMotionTpl(const FrameIndex& id, const Motion& motion, pinocchio::ReferenceFrame reference = pinocchio::LOCAL)
      : id(id), motion(motion), reference(reference) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameMotionTpl<Scalar>& X) {
    os << "       id: " << X.id << std::endl;
    os << "   motion: " << std::endl << X.motion;
    switch (X.reference) {
      case pinocchio::WORLD:
        os << "reference: WORLD" << std::endl;
        break;
      case pinocchio::LOCAL:
        os << "reference: LOCAL" << std::endl;
        break;
      case pinocchio::LOCAL_WORLD_ALIGNED:
        os << "reference: LOCAL_WORLD_ALIGNED" << std::endl;
        break;
    }
    return os;
  }

  template <typename OtherScalar>
  bool operator==(const FrameMotionTpl<OtherScalar>& other) const {
    return id == other.id && motion == other.motion && reference == other.reference;
  }

  FrameIndex id;
  pinocchio::MotionTpl<Scalar> motion;
  pinocchio::ReferenceFrame reference;
};

template <typename _Scalar>
struct FrameForceTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::ForceTpl<Scalar> Force;

  explicit FrameForceTpl() : id(0), force(Force::Zero()) {}
  FrameForceTpl(const FrameForceTpl<Scalar>& value) : id(value.id), force(value.force) {}
  FrameForceTpl(const FrameIndex& id, const Force& force) : id(id), force(force) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameForceTpl<Scalar>& X) {
    os << "   id: " << X.id << std::endl << "force: " << std::endl << X.force << std::endl;
    return os;
  }

  template <typename OtherScalar>
  bool operator==(const FrameForceTpl<OtherScalar>& other) const {
    return id == other.id && force == other.force;
  }

  FrameIndex id;
  pinocchio::ForceTpl<Scalar> force;
};

template <typename _Scalar>
struct FrameFrictionConeTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef FrictionConeTpl<Scalar> FrictionCone;

  explicit FrameFrictionConeTpl() : id(0), cone(FrictionCone()) {}
  FrameFrictionConeTpl(const FrameFrictionConeTpl<Scalar>& value) : id(value.id), cone(value.cone) {}
  FrameFrictionConeTpl(const FrameIndex& id, const FrictionCone& cone) : id(id), cone(cone) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameFrictionConeTpl& X) {
    os << "  id: " << X.id << std::endl << "cone: " << std::endl << X.cone << std::endl;
    return os;
  }

  FrameIndex id;
  FrictionCone cone;
};

template <typename _Scalar>
struct FrameWrenchConeTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef WrenchConeTpl<Scalar> WrenchCone;

  explicit FrameWrenchConeTpl() : id(0), cone(WrenchCone()) {}
  FrameWrenchConeTpl(const FrameWrenchConeTpl<Scalar>& value) : id(value.id), cone(value.cone) {}
  FrameWrenchConeTpl(const FrameIndex& id, const WrenchCone& cone) : id(id), cone(cone) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameWrenchConeTpl& X) {
    os << "frame: " << X.id << std::endl << " cone: " << std::endl << X.cone << std::endl;
    return os;
  }

  FrameIndex id;
  WrenchCone cone;
};

template <typename _Scalar>
struct FrameCoPSupportTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;

 public:
  explicit FrameCoPSupportTpl() : id_(0), box_(Vector2s::Zero()) { update_A(); }
  FrameCoPSupportTpl(const FrameCoPSupportTpl<Scalar>& value)
      : id_(value.get_id()), box_(value.get_box()), A_(value.get_A()) {}
  FrameCoPSupportTpl(const FrameIndex& id, const Vector2s& box) : id_(id), box_(box) { update_A(); }
  friend std::ostream& operator<<(std::ostream& os, const FrameCoPSupportTpl<Scalar>& X) {
    os << " id: " << X.get_id() << std::endl << "box: " << std::endl << X.get_box() << std::endl;
    return os;
  }

  // Define the inequality matrix A to implement A * f >= 0. Compare eq.(18-19) in
  // https://hal.archives-ouvertes.fr/hal-02108449/document
  void update_A() {
    A_ << Scalar(0), Scalar(0), box_[0] / Scalar(2), Scalar(0), Scalar(-1), Scalar(0), Scalar(0), Scalar(0),
        box_[0] / Scalar(2), Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0), box_[1] / Scalar(2), Scalar(1),
        Scalar(0), Scalar(0), Scalar(0), Scalar(0), box_[1] / Scalar(2), Scalar(-1), Scalar(0), Scalar(0);
  }

  void set_id(FrameIndex id) { id_ = id; }
  void set_box(const Vector2s& box) {
    box_ = box;
    update_A();
  }

  const FrameIndex& get_id() const { return id_; }
  const Vector2s& get_box() const { return box_; }
  const Matrix46& get_A() const { return A_; }

 private:
  FrameIndex id_;  //!< contact frame ID
  Vector2s box_;   //!< cop support region = (length, width)
  Matrix46 A_;     //!< inequality matrix
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

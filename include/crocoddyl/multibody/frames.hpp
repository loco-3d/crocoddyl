
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRAMES_HPP_
#define CROCODDYL_MULTIBODY_FRAMES_HPP_


CROCODDYL_PRAGMA_TO_BE_REMOVED_HEADER("crocoddyl/multibody/frames.hpp")

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/force.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

typedef std::size_t FrameIndex;

template <typename _Scalar>
struct FrameTranslationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Vector3s Vector3s;

  DEPRECATED("Do not use FrameTranslation", explicit FrameTranslationTpl());
  DEPRECATED("Do not use FrameTranslation", FrameTranslationTpl(const FrameTranslationTpl<Scalar>& other));
  DEPRECATED("Do not use FrameTranslation", FrameTranslationTpl(const FrameIndex& id, const Vector3s& translation));

  friend std::ostream& operator<<(std::ostream& os, const FrameTranslationTpl<Scalar>& X) {
    os << "         id: " << X.id << std::endl
       << "translation: " << std::endl
       << X.translation.transpose() << std::endl;
    return os;
  }

  FrameTranslationTpl<Scalar>& operator=(const FrameTranslationTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      translation = other.translation;
    }
    return *this;
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

  DEPRECATED("Do not use FrameRotation", explicit FrameRotationTpl());
  DEPRECATED("Do not use FrameRotation", FrameRotationTpl(const FrameRotationTpl<Scalar>& other));
  DEPRECATED("Do not use FrameRotation", FrameRotationTpl(const FrameIndex& id, const Matrix3s& rotation));

  friend std::ostream& operator<<(std::ostream& os, const FrameRotationTpl<Scalar>& X) {
    os << "      id: " << X.id << std::endl << "rotation: " << std::endl << X.rotation << std::endl;
    return os;
  }

  FrameRotationTpl<Scalar>& operator=(const FrameRotationTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      rotation = other.rotation;
    }
    return *this;
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

  DEPRECATED("Do not use FramePlacement", explicit FramePlacementTpl());
  DEPRECATED("Do not use FramePlacement", FramePlacementTpl(const FramePlacementTpl<Scalar>& other));
  DEPRECATED("Do not use FramePlacement", FramePlacementTpl(const FrameIndex& id, const SE3& placement));

  FramePlacementTpl<Scalar>& operator=(const FramePlacementTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      placement = other.placement;
    }
    return *this;
  }

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

  DEPRECATED("Do not use FrameMotion", explicit FrameMotionTpl());
  DEPRECATED("Do not use FrameMotion", FrameMotionTpl(const FrameMotionTpl<Scalar>& other));
  DEPRECATED("Do not use FrameMotion", FrameMotionTpl(const FrameIndex& id, const Motion& motion,
                                                      pinocchio::ReferenceFrame reference = pinocchio::LOCAL));

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

  FrameMotionTpl<Scalar>& operator=(const FrameMotionTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      motion = other.motion;
      reference = other.reference;
    }
    return *this;
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

  DEPRECATED("Do not use FrameForce", explicit FrameForceTpl());
  DEPRECATED("Do not use FrameForce", FrameForceTpl(const FrameForceTpl<Scalar>& other));
  DEPRECATED("Do not use FrameForce", FrameForceTpl(const FrameIndex& id, const Force& force));

  friend std::ostream& operator<<(std::ostream& os, const FrameForceTpl<Scalar>& X) {
    os << "   id: " << X.id << std::endl << "force: " << std::endl << X.force << std::endl;
    return os;
  }

  FrameForceTpl<Scalar>& operator=(const FrameForceTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      force = other.force;
    }
    return *this;
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

  DEPRECATED("Do not use FrameFrictionCone", explicit FrameFrictionConeTpl());
  DEPRECATED("Do not use FrameFrictionCone", FrameFrictionConeTpl(const FrameFrictionConeTpl<Scalar>& other));
  DEPRECATED("Do not use FrameFrictionCone", FrameFrictionConeTpl(const FrameIndex& id, const FrictionCone& cone));

  friend std::ostream& operator<<(std::ostream& os, const FrameFrictionConeTpl& X) {
    os << "  id: " << X.id << std::endl << "cone: " << std::endl << X.cone << std::endl;
    return os;
  }

  FrameFrictionConeTpl<Scalar>& operator=(const FrameFrictionConeTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      cone = other.cone;
    }
    return *this;
  }

  FrameIndex id;
  FrictionCone cone;
};

template <typename _Scalar>
struct FrameWrenchConeTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef WrenchConeTpl<Scalar> WrenchCone;

  DEPRECATED("Do not use FrameWrenchCone", explicit FrameWrenchConeTpl());
  DEPRECATED("Do not use FrameWrenchCone", FrameWrenchConeTpl(const FrameWrenchConeTpl<Scalar>& other));
  DEPRECATED("Do not use FrameWrenchCone", FrameWrenchConeTpl(const FrameIndex& id, const WrenchCone& cone));

  friend std::ostream& operator<<(std::ostream& os, const FrameWrenchConeTpl& X) {
    os << "frame: " << X.id << std::endl << " cone: " << std::endl << X.cone << std::endl;
    return os;
  }

  FrameWrenchConeTpl<Scalar>& operator=(const FrameWrenchConeTpl<Scalar>& other) {
    if (this != &other) {
      id = other.id;
      cone = other.cone;
    }
    return *this;
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
  DEPRECATED("Do not use FrameCoPSupport", explicit FrameCoPSupportTpl());
  DEPRECATED("Do not use FrameCoPSupport", FrameCoPSupportTpl(const FrameCoPSupportTpl<Scalar>& other));
  DEPRECATED("Do not use FrameCoPSupport", FrameCoPSupportTpl(const FrameIndex& id, const Vector2s& box));

  friend std::ostream& operator<<(std::ostream& os, const FrameCoPSupportTpl<Scalar>& X) {
    os << " id: " << X.get_id() << std::endl << "box: " << std::endl << X.get_box() << std::endl;
    return os;
  }

  FrameCoPSupportTpl<Scalar>& operator=(const FrameCoPSupportTpl<Scalar>& other) {
    if (this != &other) {
      id_ = other.get_id();
      box_ = other.get_box();
      A_ = other.get_A();
    }
    return *this;
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

template <typename Scalar>
FrameTranslationTpl<Scalar>::FrameTranslationTpl() : id(0), translation(Vector3s::Zero()) {
  std::cerr << "Deprecated: Do not use FrameTranslation." << std::endl;
}
template <typename Scalar>
FrameTranslationTpl<Scalar>::FrameTranslationTpl(const FrameTranslationTpl<Scalar>& other)
    : id(other.id), translation(other.translation) {
  std::cerr << "Deprecated: Do not use FrameTranslation." << std::endl;
}
template <typename Scalar>
FrameTranslationTpl<Scalar>::FrameTranslationTpl(const FrameIndex& id, const Vector3s& translation)
    : id(id), translation(translation) {
  std::cerr << "Deprecated: Do not use FrameTranslation." << std::endl;
}

template <typename Scalar>
FrameRotationTpl<Scalar>::FrameRotationTpl() : id(0), rotation(Matrix3s::Identity()) {
  std::cerr << "Deprecated: Do not use FrameRotation." << std::endl;
}
template <typename Scalar>
FrameRotationTpl<Scalar>::FrameRotationTpl(const FrameRotationTpl<Scalar>& other)
    : id(other.id), rotation(other.rotation) {
  std::cerr << "Deprecated: Do not use FrameRotation." << std::endl;
}
template <typename Scalar>
FrameRotationTpl<Scalar>::FrameRotationTpl(const FrameIndex& id, const Matrix3s& rotation)
    : id(id), rotation(rotation) {
  std::cerr << "Deprecated: Do not use FrameRotation." << std::endl;
}

template <typename Scalar>
FramePlacementTpl<Scalar>::FramePlacementTpl() : id(0), placement(SE3::Identity()) {
  std::cerr << "Deprecated: Do not use FramePlacement." << std::endl;
}
template <typename Scalar>
FramePlacementTpl<Scalar>::FramePlacementTpl(const FramePlacementTpl<Scalar>& other)
    : id(other.id), placement(other.placement) {
  std::cerr << "Deprecated: Do not use FramePlacement." << std::endl;
}
template <typename Scalar>
FramePlacementTpl<Scalar>::FramePlacementTpl(const FrameIndex& id, const SE3& placement)
    : id(id), placement(placement) {
  std::cerr << "Deprecated: Do not use FramePlacement." << std::endl;
}

template <typename Scalar>
FrameMotionTpl<Scalar>::FrameMotionTpl() : id(0), motion(Motion::Zero()), reference(pinocchio::LOCAL) {
  std::cerr << "Deprecated: Do not use FrameMotion." << std::endl;
}
template <typename Scalar>
FrameMotionTpl<Scalar>::FrameMotionTpl(const FrameMotionTpl<Scalar>& other)
    : id(other.id), motion(other.motion), reference(other.reference) {
  std::cerr << "Deprecated: Do not use FrameMotion." << std::endl;
}
template <typename Scalar>
FrameMotionTpl<Scalar>::FrameMotionTpl(const FrameIndex& id, const Motion& motion, pinocchio::ReferenceFrame reference)
    : id(id), motion(motion), reference(reference) {
  std::cerr << "Deprecated: Do not use FrameMotion." << std::endl;
}

template <typename Scalar>
FrameForceTpl<Scalar>::FrameForceTpl() : id(0), force(Force::Zero()) {
  std::cerr << "Deprecated: Do not use FrameForce." << std::endl;
}
template <typename Scalar>
FrameForceTpl<Scalar>::FrameForceTpl(const FrameForceTpl<Scalar>& other) : id(other.id), force(other.force) {
  std::cerr << "Deprecated: Do not use FrameForce." << std::endl;
}
template <typename Scalar>
FrameForceTpl<Scalar>::FrameForceTpl(const FrameIndex& id, const Force& force) : id(id), force(force) {
  std::cerr << "Deprecated: Do not use FrameForce." << std::endl;
}

template <typename Scalar>
FrameFrictionConeTpl<Scalar>::FrameFrictionConeTpl() : id(0), cone(FrictionCone()) {
  std::cerr << "Deprecated: Do not use FrameFrictionCone." << std::endl;
}
template <typename Scalar>
FrameFrictionConeTpl<Scalar>::FrameFrictionConeTpl(const FrameFrictionConeTpl<Scalar>& other)
    : id(other.id), cone(other.cone) {
  std::cerr << "Deprecated: Do not use FrameFrictionCone." << std::endl;
}
template <typename Scalar>
FrameFrictionConeTpl<Scalar>::FrameFrictionConeTpl(const FrameIndex& id, const FrictionCone& cone)
    : id(id), cone(cone) {
  std::cerr << "Deprecated: Do not use FrameFrictionCone." << std::endl;
}

template <typename Scalar>
FrameWrenchConeTpl<Scalar>::FrameWrenchConeTpl() : id(0), cone(WrenchCone()) {
  std::cerr << "Deprecated: Do not use FrameWrenchCone." << std::endl;
}
template <typename Scalar>
FrameWrenchConeTpl<Scalar>::FrameWrenchConeTpl(const FrameWrenchConeTpl<Scalar>& other)
    : id(other.id), cone(other.cone) {
  std::cerr << "Deprecated: Do not use FrameWrenchCone." << std::endl;
}
template <typename Scalar>
FrameWrenchConeTpl<Scalar>::FrameWrenchConeTpl(const FrameIndex& id, const WrenchCone& cone) : id(id), cone(cone) {
  std::cerr << "Deprecated: Do not use FrameWrenchCone." << std::endl;
}

template <typename Scalar>
FrameCoPSupportTpl<Scalar>::FrameCoPSupportTpl() : id_(0), box_(Vector2s::Zero()) {
  update_A();
  std::cerr << "Deprecated: Do not use FrameCoPSupport." << std::endl;
}
template <typename Scalar>
FrameCoPSupportTpl<Scalar>::FrameCoPSupportTpl(const FrameCoPSupportTpl<Scalar>& other)
    : id_(other.get_id()), box_(other.get_box()), A_(other.get_A()) {
  std::cerr << "Deprecated: Do not use FrameCoPSupport." << std::endl;
}
template <typename Scalar>
FrameCoPSupportTpl<Scalar>::FrameCoPSupportTpl(const FrameIndex& id, const Vector2s& box) : id_(id), box_(box) {
  update_A();
  std::cerr << "Deprecated: Do not use FrameCoPSupport." << std::endl;
}

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_

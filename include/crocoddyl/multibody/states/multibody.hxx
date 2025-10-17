///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <limits>
#include <vector>

namespace crocoddyl {

template <typename Scalar>
StateMultibodyTpl<Scalar>::StateMultibodyTpl(
    std::shared_ptr<PinocchioModel> model)
    : Base(model->nq + model->nv, 2 * model->nv),
      pinocchio_(model),
      x0_(VectorXs::Zero(model->nq + model->nv)) {
  x0_.head(nq_) = pinocchio::neutral(*pinocchio_.get());
  // In a multibody system, we could define the first joint using Lie groups.
  // The current cases are free-flyer (SE3) and spherical (S03).
  // Instead simple represents any joint that can model within the Euclidean
  // manifold. The rest of joints use Euclidean algebra. We use this fact for
  // computing Jdiff.
  // Define internally the limits of the first joint
  const std::size_t nq0 =
      model->existJointName("root_joint")
          ? model->joints[model->getJointId("root_joint")].nq()
          : 0;
  lb_.head(nq0) = -std::numeric_limits<Scalar>::max() * VectorXs::Ones(nq0);
  ub_.head(nq0) = std::numeric_limits<Scalar>::max() * VectorXs::Ones(nq0);
  lb_.segment(nq0, nq_ - nq0) = pinocchio_->lowerPositionLimit.tail(nq_ - nq0);
  ub_.segment(nq0, nq_ - nq0) = pinocchio_->upperPositionLimit.tail(nq_ - nq0);
  for (pinocchio::JointIndex jid = 1; jid < pinocchio_->joints.size(); ++jid) {
    const auto& j = pinocchio_->joints[jid];
    const std::size_t i = j.idx_q(), nqj = j.nq(), nvj = j.nv();
    if (nqj == 2 && nvj == 1) {  // continuous joint
      lb_[i] = -std::numeric_limits<Scalar>::max();
      lb_[i + 1] = -std::numeric_limits<Scalar>::max();
      ub_[i] = std::numeric_limits<Scalar>::max();
      ub_[i + 1] = std::numeric_limits<Scalar>::max();
    }
  }
  lb_.tail(nv_) = -pinocchio_->velocityLimit;
  ub_.tail(nv_) = pinocchio_->velocityLimit;
  Base::update_has_limits();
}

template <typename Scalar>
StateMultibodyTpl<Scalar>::StateMultibodyTpl()
    : Base(), x0_(VectorXs::Zero(0)) {}

template <typename Scalar>
StateMultibodyTpl<Scalar>::~StateMultibodyTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyTpl<Scalar>::zero() const {
  return x0_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyTpl<Scalar>::rand() const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(*pinocchio_.get());
  return xrand;
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0,
                                     const Eigen::Ref<const VectorXs>& x1,
                                     Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: x0 has wrong dimension (it should be " +
                 std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: x1 has wrong dimension (it should be " +
                 std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty("Invalid argument: dxout has wrong dimension (it should be " +
                 std::to_string(ndx_) + ")");
  }

  const Scalar MAXV = std::numeric_limits<Scalar>::max();
  auto is_invalid = [&](Scalar v) {
    return !std::isfinite(v) || v == MAXV || v == -MAXV;
  };

  VectorXs q0_clean = x0.head(nq_);
  VectorXs q1_clean = x1.head(nq_);
  VectorXs v0_clean = x0.tail(nv_);
  VectorXs v1_clean = x1.tail(nv_);

  std::vector<bool> invalid_cfg_tangent(nv_, false);
  std::vector<bool> invalid_vel_component(nv_, false);

  for (pinocchio::JointIndex jid = 1; jid < pinocchio_->joints.size(); ++jid) {
    const auto& j = pinocchio_->joints[jid];
    const std::size_t iq = j.idx_q();
    const std::size_t nqj = j.nq();
    const std::size_t iv = j.idx_v();
    const std::size_t nvj = j.nv();

    const bool quat_joint = (nqj == nvj + 1) && (nvj == 3 || nvj == 6);
    const bool freeflyer = quat_joint && (nvj == 6);

    if (quat_joint) {
      // Config layout: FF q = [t(3), quat(4 as x,y,z,w)], Spherical q = [quat(4
      // as x,y,z,w)]
      const std::size_t iq_quat =
          freeflyer ? (iq + 3) : iq;  // start of [x,y,z,w]
      // Tangent layout in Crocoddyl for FF: [linear(3), angular(3)]
      const std::size_t iv_lin = freeflyer ? (iv + 0) : iv;  // linear (FF only)
      const std::size_t iv_ang = freeflyer ? (iv + 3) : iv;  // angular

      bool quat_invalid = false;
      for (std::size_t k = 0; k < 4; ++k) {
        quat_invalid = quat_invalid || is_invalid(q0_clean[iq_quat + k]) ||
                       is_invalid(q1_clean[iq_quat + k]);
      }

      bool trans_invalid = false;
      if (freeflyer) {
        for (std::size_t k = 0; k < 3; ++k) {
          trans_invalid = trans_invalid || is_invalid(q0_clean[iq + k]) ||
                          is_invalid(q1_clean[iq + k]);
        }
      }

      if (quat_invalid) {
        // Write identity quaternion in Pinocchio order [x,y,z,w] = [0,0,0,1]
        q0_clean.segment(iq_quat, 4) << Scalar(0), Scalar(0), Scalar(0),
            Scalar(1);
        q1_clean.segment(iq_quat, 4) << Scalar(0), Scalar(0), Scalar(0),
            Scalar(1);
        // Mask ONLY the angular tangent components
        for (std::size_t k = 0; k < 3; ++k)
          invalid_cfg_tangent[iv_ang + k] = true;
      }

      if (freeflyer && trans_invalid) {
        // Neutralize translation contribution
        q0_clean.segment(iq, 3) = q1_clean.segment(iq, 3);
        for (std::size_t k = 0; k < 3; ++k)
          invalid_cfg_tangent[iv_lin + k] = true;
      }

    } else {
      bool joint_invalid = false;
      for (std::size_t k = 0; k < nqj; ++k) {
        joint_invalid = joint_invalid || is_invalid(q0_clean[iq + k]) ||
                        is_invalid(q1_clean[iq + k]);
      }
      if (joint_invalid) {
        q0_clean.segment(iq, nqj) = q1_clean.segment(iq, nqj);
        for (std::size_t k = 0; k < nvj; ++k)
          invalid_cfg_tangent[iv + k] = true;
      }
    }

    // Velocity mask per component
    for (std::size_t k = 0; k < nvj; ++k) {
      const std::size_t idx = iv + k;
      if (is_invalid(v0_clean[idx]) || is_invalid(v1_clean[idx])) {
        v0_clean[idx] = v1_clean[idx];      // neutralize
        invalid_vel_component[idx] = true;  // mask later
      }
    }
  }

  pinocchio::difference(*pinocchio_.get(), q0_clean, q1_clean, dxout.head(nv_));
  dxout.tail(nv_) = v1_clean - v0_clean;

  for (std::size_t i = 0; i < nv_; ++i) {
    if (invalid_cfg_tangent[i]) dxout[i] = MAXV;
    if (invalid_vel_component[i]) dxout[nv_ + i] = MAXV;
  }
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x,
                                          const Eigen::Ref<const VectorXs>& dx,
                                          Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty(
        "Invalid argument: " << "dx has wrong dimension (it should be " +
                                    std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "xout has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }

  pinocchio::integrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                       xout.head(nq_));
  xout.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0,
                                      const Eigen::Ref<const VectorXs>& x1,
                                      Eigen::Ref<MatrixXs> Jfirst,
                                      Eigen::Ref<MatrixXs> Jsecond,
                                      const Jcomponent firstsecond) const {
  assert_pretty(
      is_a_Jcomponent(firstsecond),
      ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x1 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }

  if (firstsecond == first) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jfirst has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }

    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_),
                           Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)-1;
  } else if (firstsecond == second) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jsecond has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_),
                           Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1);
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
  } else {  // computing both
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jfirst has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jsecond has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_),
                           Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0);
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_),
                           Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)-1;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
  }
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x,
                                           const Eigen::Ref<const VectorXs>& dx,
                                           Eigen::Ref<MatrixXs> Jfirst,
                                           Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond,
                                           const AssignmentOp op) const {
  assert_pretty(
      is_a_Jcomponent(firstsecond),
      ("firstsecond must be one of the Jcomponent {both, first, second}"));
  assert_pretty(is_a_AssignmentOp(op),
                ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jfirst has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0,
                              pinocchio::SETTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0,
                              pinocchio::ADDTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0,
                              pinocchio::RMTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "Jsecond has wrong dimension (it should be " +
                                      std::to_string(ndx_) + "," +
                                      std::to_string(ndx_) + ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1,
                              pinocchio::SETTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1,
                              pinocchio::ADDTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_),
                              Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1,
                              pinocchio::RMTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::JintegrateTransport(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
    Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const {
  assert_pretty(
      is_a_Jcomponent(firstsecond),
      ("firstsecond must be one of the Jcomponent {both, first, second}"));

  switch (firstsecond) {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_),
                                     dx.head(nv_), Jin.topRows(nv_),
                                     pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_),
                                     dx.head(nv_), Jin.topRows(nv_),
                                     pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both "
          "not supported for this operation.");
      break;
  }
}

template <typename Scalar>
const std::shared_ptr<pinocchio::ModelTpl<Scalar> >&
StateMultibodyTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
template <typename NewScalar>
StateMultibodyTpl<NewScalar> StateMultibodyTpl<Scalar>::cast() const {
  typedef StateMultibodyTpl<NewScalar> ReturnType;
  typedef pinocchio::ModelTpl<NewScalar> ModelType;
  ReturnType res(
      std::make_shared<ModelType>(pinocchio_->template cast<NewScalar>()));
  return res;
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::print(std::ostream& os) const {
  os << "StateMultibody {nx=" << nx_ << ", ndx=" << ndx_
     << ", pinocchio=" << *pinocchio_.get() << "}";
}

}  // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include "crocoddyl/core/constraints/residual.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::DifferentialActionModelFreeInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs)
    : Base(state, state->get_nv(), costs->get_nr(), 0, state->get_nv() - actuation->get_nu()),
      actuation_(actuation),
      costs_(costs),
      constraints_(boost::make_shared<ConstraintModelManager>(state, state->get_nv())),
      pinocchio_(*state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::DifferentialActionModelFreeInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs, boost::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, state->get_nv(), costs->get_nr(), constraints->get_ng(),
           constraints->get_nh() + state->get_nv() - actuation->get_nu()),
      actuation_(actuation),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::init(const boost::shared_ptr<StateMultibody>& state) {
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (constraints_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Constraints doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  VectorXs lb = VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub = VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);

  if (state->get_nv() - actuation_->get_nu() > 0) {
    constraints_->addConstraint(
        "tau",
        boost::make_shared<ConstraintModelResidual>(
            state_,
            boost::make_shared<typename DifferentialActionModelFreeInvDynamicsTpl<Scalar>::ResidualModelActuation>(
                state, actuation_->get_nu())));
  }
  constraints_->shareDimensions(this);
}

template <typename Scalar>
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::~DifferentialActionModelFreeInvDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  d->xout = u;
  pinocchio::rnea(pinocchio_, d->pinocchio, q, v, u);
  pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
  actuation_->commands(d->multibody.actuation, x, d->pinocchio.tau);
  d->multibody.joint->a = u;
  d->multibody.joint->tau = d->multibody.actuation->u;
  actuation_->calc(d->multibody.actuation, x, d->multibody.joint->tau);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  d->constraints->resize(this, d);
  constraints_->calc(d->constraints, x, u);
}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);
  d->constraints->resize(this, d);

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, u);
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.template triangularView<Eigen::StrictlyUpper>().transpose();
  actuation_->calcDiff(d->multibody.actuation, x, d->multibody.joint->tau);
  actuation_->torqueTransform(d->multibody.actuation, x, d->multibody.joint->tau);
  d->multibody.joint->dtau_dx.leftCols(nv).noalias() = d->multibody.actuation->Mtau * d->pinocchio.dtau_dq;
  d->multibody.joint->dtau_dx.rightCols(nv).noalias() = d->multibody.actuation->Mtau * d->pinocchio.dtau_dv;
  d->multibody.joint->dtau_du.noalias() = d->multibody.actuation->Mtau * d->pinocchio.M;
  costs_->calcDiff(d->costs, x, u);
  constraints_->calcDiff(d->constraints, x, u);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelFreeInvDynamicsTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>&, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>&, const std::size_t, const Scalar) {
  u.setZero();
}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelFreeInvDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
     << ", nu=" << nu_ << "}";
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

}  // namespace crocoddyl

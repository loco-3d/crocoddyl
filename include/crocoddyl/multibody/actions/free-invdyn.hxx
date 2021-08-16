///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actions/free-invdyn.hpp"

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/cholesky.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::DifferentialActionModelFreeInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs, boost::shared_ptr<ConstraintModelManager> constraints)
    :  
    // if (constraints_ != nullptr) {
    //   Base(state, state.get_nv()+actuation->get_nu(), costs->get_nr()), constraints.get_ng(), constraints.get_nh()+state.get_nv()),
    //   constraints_(constraints),
    // }
    //if (constraints_ == nullptr){
      Base(state, state.get_nv()+actuation->get_nu(), costs->get_nr()), 0, constraints.get_nh()+state.get_nv()),
      constraints_(ConstraintModelManager(state, state->get_nv() + actuation->get_nu())),
    //}
   
    
      actuation_(actuation),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get())) 
      
      {
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));

  if (constraints_ != nullptr) {
    ng_ = constraints_->get_ng();
    nh_ = constraints_->get_nh();
  }
  constraints_->addConstraint(
      "rnea", ConstraintModelResidual(
                  state_, DifferentialActionModelFreeInvDynamics::ResidualModelRnea(state_, actuation_->get_nu())))
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
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = x.head(state_->get_nv());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = x.tail(state_->get_nv());

  d->xout = a pinocchio::rnea(pinocchio_, d->pinocchio, q, v, a);
  pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  actuation_->calc(d->multibody.actuation, x, u);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    constraints_->calc(d->constraints, x, u);
  }
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
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  actuation_->calcDiff(d->multibody.actuation, x, u);

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout);

  // Computing the cost derivatives
  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
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
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t, const Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

  // Check the velocity input is zero
  assert_pretty(x.tail(state_->get_nv()).isZero(), "The velocity input should be zero for quasi-static to work.");

  d->tmp_xstatic.head(state_->get_nq()) = q;
  d->tmp_xstatic.tail(state_->get_nq()) *= 0;
  d->tmp_ustatic.setZero();

  pinocchio::rnea(pinocchio_, d->pinocchio, q, VectorXs::Zero(state_->get_nv()), VectorXs::Zero(state_->get_nv()));

  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, VectorXs::Zero(nu_));
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, VectorXs::Zero(nu_));

  u.noalias() = pseudoInverse(d->multibody.actuation->dtau_du) * d->pinocchio.tau;
  d->pinocchio.tau.setZero();
  return d->tmp_ustatic
}

template <typename Scalar>
void DifferentialActionModelFreeInvDynamicsTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelFreeFwdDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
     << ", nu=" << nu_ << "}";
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar>>&
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar>>& DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelManagerTpl<Scalar>>&
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::ResidualModelRneaTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  d->r = d->pinocchio->tau - d->actuation->tau
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::ResidualModelRneaTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  d->Rx.leftCols(nv) = d->pinocchio->dtau_dq d->Rx.rightCols(nv) = d->pinocchio->dtau_dv d->Rx -=
      d->shared.actuation->dtau_dx d->Ru.leftCols(nv) = d->pinocchio->M d->Ru.rightCols(nv) = d->actuation->dtau_du
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar>>
DifferentialActionModelFreeInvDynamicsTpl<Scalar>::ResidualModelRneaTpl<Scalar>::createData() {
  return boost::allocate_shared<rneaData>(Eigen::aligned_allocator<rneaData>(), this);
}

// template <typename Scalar>
// bool DifferentialActionModelFreeInvDynamicsTpl<Scalar>::ResidualModelRneaTpl<Scalar>::checkData(
//     const boost::shared_ptr<ResidualDataAbstract>& data) {
//   boost::shared_ptr<rneaData> d = boost::dynamic_pointer_cast<rneaData>(data);
//   if (d != NULL) {
//     return true;
//   } else {
//     return false;
//   }
// }

}  // namespace crocoddyl

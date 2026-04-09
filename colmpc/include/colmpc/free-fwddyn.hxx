///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/utils/exception.hpp>
#include <pinocchio/collision/distance.hpp>

#include "colmpc/free-fwddyn.hpp"

namespace colmpc {

template <typename Scalar>
DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::
    DifferentialActionModelFreeFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, actuation, costs, constraints),
      geometry_(*state->get_geometry().get()) {}

template <typename Scalar>
DifferentialActionModelFreeFwdDynamicsTpl<
    Scalar>::~DifferentialActionModelFreeFwdDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  pinocchio::computeDistances(this->get_pinocchio(), d->pinocchio, geometry_,
                              d->geometry, q);
  Base::calc(data, x, u);
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  pinocchio::computeDistances(this->get_pinocchio(), d->pinocchio, geometry_,
                              d->geometry, q);
  Base::calc(data, x);
}

template <typename Scalar>
std::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<Scalar>>
DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelFreeFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", ncolpairs=" << geometry_.collisionPairs.size() << "}";
}

template <typename Scalar>
DifferentialActionDataFreeFwdDynamicsTpl<Scalar>::
    DifferentialActionDataFreeFwdDynamicsTpl(
        DifferentialActionModelFreeFwdDynamicsTpl<Scalar>* const model)
    : Base(model), geometry(model->get_geometry()) {
  for (auto& name_cost : this->costs->costs) {
    auto w = std::dynamic_pointer_cast<GeometryDataWrapper>(
        name_cost.second->residual);
    if (w) {
      w->setGeometry(&geometry);
    }
  }

  for (auto& name_constraints : this->constraints->constraints) {
    auto w = std::dynamic_pointer_cast<GeometryDataWrapper>(
        name_constraints.second->residual);
    if (w) {
      w->setGeometry(&geometry);
    }
  }
}

}  // namespace colmpc

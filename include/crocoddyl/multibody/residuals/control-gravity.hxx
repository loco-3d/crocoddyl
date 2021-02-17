///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelControlGravTpl<Scalar>::ResidualModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                                 const std::size_t nu)
    : Base(state, state->get_nv(), nu), pin_model_(*state->get_pinocchio()) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
ResidualModelControlGravTpl<Scalar>::ResidualModelControlGravTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, state->get_nv(), state->get_nv()), pin_model_(*state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelControlGravTpl<Scalar>::~ResidualModelControlGravTpl() {}

template <typename Scalar>
void ResidualModelControlGravTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                               const Eigen::Ref<const VectorXs> &x,
                                               const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  data->r = d->actuation->tau - pinocchio::computeGeneralizedGravity(pin_model_, d->pinocchio, q);
}

template <typename Scalar>
void ResidualModelControlGravTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                   const Eigen::Ref<const VectorXs> &x,
                                                   const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the derivatives of the residual residual
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const std::size_t nv = state_->get_nv();
  Eigen::Ref<MatrixXs> Rq(data->Rx.leftCols(nv));
  pinocchio::computeGeneralizedGravityDerivatives(pin_model_, d->pinocchio, q, Rq);
  Rq *= -1;
  data->Ru = d->actuation->dtau_du;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelControlGravTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl

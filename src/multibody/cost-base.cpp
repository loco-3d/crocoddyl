///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

CostModelAbstract::CostModelAbstract(boost::shared_ptr<StateMultibody> state, ActivationModelAbstract& activation, const std::size_t& nu,
                                     const bool& with_residuals)
    : state_(state),
      activation_(activation),
      nu_(nu),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(nu)) {}

CostModelAbstract::CostModelAbstract(boost::shared_ptr<StateMultibody> state, ActivationModelAbstract& activation,
                                     const bool& with_residuals)
    : state_(state),
      activation_(activation),
      nu_(state->get_nv()),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(state->get_nv())) {}

CostModelAbstract::CostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr, const std::size_t& nu,
                                     const bool& with_residuals)
    : state_(state),
      activation_(*new ActivationModelQuad(nr)),
      nu_(nu),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(nu)) {}

CostModelAbstract::CostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr, const bool& with_residuals)
    : state_(state),
      activation_(*new ActivationModelQuad(nr)),
      nu_(state->get_nv()),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(state->get_nv())) {}

CostModelAbstract::~CostModelAbstract() {}

void CostModelAbstract::calc(const boost::shared_ptr<CostDataAbstract>& data,
                             const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void CostModelAbstract::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                 const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

boost::shared_ptr<CostDataAbstract> CostModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataAbstract>(this, data);
}

const boost::shared_ptr<StateMultibody>& CostModelAbstract::get_state() const { return state_; }

ActivationModelAbstract& CostModelAbstract::get_activation() const { return activation_; }

const std::size_t& CostModelAbstract::get_nu() const { return nu_; }

}  // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelControlGrav>(state, nu)) {
  std::cerr << "Deprecated CostModelControlGrav: Use ResidualModelControlGrav with CostModelResidual" << std::endl;
  if (activation_->get_nr() != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_nv()));
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelControlGrav>(state)) {
  std::cerr << "Deprecated CostModelControlGrav: Use ResidualModelControlGrav with CostModelResidual" << std::endl;
  if (activation_->get_nr() != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_nv()));
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelControlGrav>(state, nu)) {
  std::cerr << "Deprecated CostModelControlGrav: Use ResidualModelControlGrav with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, boost::make_shared<ResidualModelControlGrav>(state)) {
  std::cerr << "Deprecated CostModelControlGrav: Use ResidualModelControlGrav with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::~CostModelControlGravTpl() {}

}  // namespace crocoddyl

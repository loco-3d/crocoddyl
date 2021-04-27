///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state,
                                                       boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelImpulseCoM>(state)) {
  std::cerr << "Deprecated CostModelImpulseCoM: Use ResidualModelImpulseCoM with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, boost::make_shared<ResidualModelImpulseCoM>(state)) {
  std::cerr << "Deprecated CostModelImpulseCoM: Use ResidualModelImpulseCoM with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::~CostModelImpulseCoMTpl() {}

}  // namespace crocoddyl

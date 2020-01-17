///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulse-cost-base.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

ImpulseCostModelAbstract::ImpulseCostModelAbstract(boost::shared_ptr<StateMultibody> state,
                                                   boost::shared_ptr<ActivationModelAbstract> activation,
                                                   const bool& with_residuals)
    : state_(state), activation_(activation), with_residuals_(with_residuals) {}

ImpulseCostModelAbstract::ImpulseCostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr,
                                                   const bool& with_residuals)
    : state_(state), activation_(boost::make_shared<ActivationModelQuad>(nr)), with_residuals_(with_residuals) {}

ImpulseCostModelAbstract::~ImpulseCostModelAbstract() {}

boost::shared_ptr<ImpulseCostDataAbstract> ImpulseCostModelAbstract::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<ImpulseCostDataAbstract>(this, data);
}

const boost::shared_ptr<StateMultibody>& ImpulseCostModelAbstract::get_state() const { return state_; }

const boost::shared_ptr<ActivationModelAbstract>& ImpulseCostModelAbstract::get_activation() const {
  return activation_;
}

}  // namespace crocoddyl

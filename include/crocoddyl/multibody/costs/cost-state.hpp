///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_COST_STATE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_COST_STATE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
// TODO: CostModelControl CostDataControl

namespace crocoddyl {

class CostModelState : public CostModelAbstract {
 public:
  CostModelState(pinocchio::Model* const model, StateAbstract* state, ActivationModelAbstract* const activation,
                 const Eigen::Ref<const Eigen::VectorXd>& xref, const unsigned int& nu);
  CostModelState(pinocchio::Model* const model, StateAbstract* state, const Eigen::Ref<const Eigen::VectorXd>& xref,
                 const unsigned int& nu);
  ~CostModelState();

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

 private:
  StateAbstract* state_;
  Eigen::VectorXd xref_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_COST_CONTROL_HPP_

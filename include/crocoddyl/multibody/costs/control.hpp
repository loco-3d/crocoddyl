///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelControl : public CostModelAbstract {
 public:
  CostModelControl(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                   const Eigen::VectorXd& uref, const unsigned int& nu);
  CostModelControl(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                   const Eigen::VectorXd& uref);
  CostModelControl(pinocchio::Model* const model, const Eigen::VectorXd& uref, const unsigned int& nu);
  CostModelControl(pinocchio::Model* const model, const Eigen::VectorXd& uref);
  CostModelControl(pinocchio::Model* const model, ActivationModelAbstract* const activation, const unsigned int& nu);
  CostModelControl(pinocchio::Model* const model, const unsigned int& nu);
  CostModelControl(pinocchio::Model* const model, ActivationModelAbstract* const activation);
  CostModelControl(pinocchio::Model* const model);

  ~CostModelControl();

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

 private:
  Eigen::VectorXd uref_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

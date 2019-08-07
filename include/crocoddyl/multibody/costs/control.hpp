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
  CostModelControl(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::VectorXd& uref);
  CostModelControl(StateMultibody& state, ActivationModelAbstract& activation);
  CostModelControl(StateMultibody& state, ActivationModelAbstract& activation, const unsigned int& nu);
  CostModelControl(StateMultibody& state, const Eigen::VectorXd& uref);
  CostModelControl(StateMultibody& state);
  CostModelControl(StateMultibody& state, const unsigned int& nu);
  ~CostModelControl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

  const Eigen::VectorXd& get_uref() const;

 private:
  Eigen::VectorXd uref_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

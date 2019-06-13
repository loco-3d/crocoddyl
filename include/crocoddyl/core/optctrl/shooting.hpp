///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCtRL_SHOOTING_HPP_

#include <crocoddyl/core/action-base.hpp>

namespace crocoddyl {

class ShootingProblem {
 public:
  ShootingProblem(const Eigen::Ref<const Eigen::VectorXd>& x0,
                  std::vector<ActionModelAbstract*>& runningModels,
                  ActionModelAbstract* terminalModel);
  ~ShootingProblem();

  double calc(const std::vector<Eigen::VectorXd>& xs,
              const std::vector<Eigen::VectorXd>& us);
  double calcDiff(const std::vector<Eigen::VectorXd>& xs,
                  const std::vector<Eigen::VectorXd>& us);
  void rollout(const std::vector<Eigen::VectorXd>& us,
               std::vector<Eigen::VectorXd>& xs);

  long unsigned int get_T() const;
  Eigen::VectorXd& get_x0();

  ActionModelAbstract* terminalModel;
  std::shared_ptr<ActionDataAbstract> terminalData;
  std::vector<ActionModelAbstract*> runningModels;
  std::vector<std::shared_ptr<ActionDataAbstract>> runningDatas;

 protected:
  long unsigned int T;
  Eigen::VectorXd x0;

 private:
  double cost;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

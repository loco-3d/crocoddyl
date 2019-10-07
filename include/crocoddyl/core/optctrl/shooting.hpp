///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

#include <vector>
#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

class ShootingProblem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ShootingProblem(const Eigen::VectorXd& x0, const std::vector<ActionModelAbstract*>& running_models,
                  ActionModelAbstract* const terminal_model);
  ~ShootingProblem();

  double calc(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);
  double calcDiff(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);
  void rollout(const std::vector<Eigen::VectorXd>& us, std::vector<Eigen::VectorXd>& xs);
  std::vector<Eigen::VectorXd> rollout_us(const std::vector<Eigen::VectorXd>& us);

  const std::size_t& get_T() const;
  const Eigen::VectorXd& get_x0() const;

  std::vector<ActionModelAbstract*>& get_runningModels();
  ActionModelAbstract* get_terminalModel();
  std::vector<boost::shared_ptr<ActionDataAbstract> >& get_runningDatas();
  boost::shared_ptr<ActionDataAbstract>& get_terminalData();

  ActionModelAbstract* terminal_model_;
  boost::shared_ptr<ActionDataAbstract> terminal_data_;
  std::vector<ActionModelAbstract*> running_models_;
  std::vector<boost::shared_ptr<ActionDataAbstract> > running_datas_;

 protected:
  void allocateData();
  std::size_t T_;
  Eigen::VectorXd x0_;

 private:
  double cost_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

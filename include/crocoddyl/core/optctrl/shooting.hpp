///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
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

  ShootingProblem(const Eigen::VectorXd& x0,
                  const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
                  boost::shared_ptr<ActionModelAbstract> terminal_model);
  ~ShootingProblem();

  double calc(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);
  double calcDiff(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);
  void rollout(const std::vector<Eigen::VectorXd>& us, std::vector<Eigen::VectorXd>& xs);
  std::vector<Eigen::VectorXd> rollout_us(const std::vector<Eigen::VectorXd>& us);

  const std::size_t& get_T() const;
  const Eigen::VectorXd& get_x0() const;
  void set_x0(Eigen::VectorXd x0_in);

  const std::vector<boost::shared_ptr<ActionModelAbstract> >& get_runningModels() const;
  const boost::shared_ptr<ActionModelAbstract>& get_terminalModel() const;
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& get_runningDatas() const;
  const boost::shared_ptr<ActionDataAbstract>& get_terminalData() const;

 protected:
  double cost_;
  std::size_t T_;
  Eigen::VectorXd x0_;
  boost::shared_ptr<ActionModelAbstract> terminal_model_;
  boost::shared_ptr<ActionDataAbstract> terminal_data_;
  std::vector<boost::shared_ptr<ActionModelAbstract> > running_models_;
  std::vector<boost::shared_ptr<ActionDataAbstract> > running_datas_;

 private:
  void allocateData();
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

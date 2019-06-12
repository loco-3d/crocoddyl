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
                  ActionModelAbstract* terminalModel) :terminalModel(terminalModel),
      runningModels(runningModels), T(runningModels.size()), x0(x0), cost(0.) {
    for (unsigned int i = 0; i < runningModels.size(); ++i) {
      ActionModelAbstract* model = runningModels[i];
      runningDatas.push_back(model->createData());
    }
    terminalData = terminalModel->createData();
  }
  ~ShootingProblem() { }

  double calc(const std::vector<Eigen::VectorXd>& xs,
              const std::vector<Eigen::VectorXd>& us) {
    double cost = 0;
    for (unsigned int i = 0; i < T; ++i) {
      ActionModelAbstract* model = runningModels[i];
      std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
      const Eigen::VectorXd& x = xs[i+1];
      const Eigen::VectorXd& u = us[i];

      model->calc(data, x, u);
      cost += data->cost;
    }
    terminalModel->calc(terminalData, xs.back());
    cost += terminalData->cost;
    return cost;
  }

  double calcDiff(const std::vector<Eigen::VectorXd>& xs,
                  const std::vector<Eigen::VectorXd>& us) {
    cost = 0;
    for (long unsigned int i = 0; i < T; ++i) {
      ActionModelAbstract* model = runningModels[i];
      std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
      const Eigen::VectorXd& x = xs[i];
      const Eigen::VectorXd& u = us[i];

      model->calcDiff(data, x, u);
      cost += data->cost;
    }
    terminalModel->calcDiff(terminalData, xs.back());
    cost += terminalData->cost;
    return cost;
  }

  void rollout(const std::vector<Eigen::VectorXd>& us,
               std::vector<Eigen::VectorXd>& xs) {
    xs.resize(T+1);
    xs[0] = x0;
    for (long unsigned int i = 0; i < T; ++i) {
      ActionModelAbstract* model = runningModels[i];
      std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
      Eigen::VectorXd& x = xs[i];
      const Eigen::VectorXd& u = us[i];

      model->calc(data, x, u);
      xs[i+1] = data->get_xnext();
    }
  }

  long unsigned int get_T() const { return T; }
  Eigen::VectorXd& get_x0() { return x0; }

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

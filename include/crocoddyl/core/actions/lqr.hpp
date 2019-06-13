///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/states/state-euclidean.hpp>

//TODO: DifferentialActionModelLQR DifferentialActionDataLQR

namespace crocoddyl {

class ActionModelLQR : public ActionModelAbstract {
 public:
  ActionModelLQR(const unsigned int& nx,
                 const unsigned int& nu,
                 bool driftFree=true);
  ~ActionModelLQR();

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override;
  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc=true) override;
  std::shared_ptr<ActionDataAbstract> createData() override;

  Eigen::MatrixXd Fx;
  Eigen::MatrixXd Fu;
  Eigen::VectorXd f0;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
  Eigen::VectorXd lx;
  Eigen::VectorXd lu;

 private:
  bool driftFree;
};

struct ActionDataLQR : public ActionDataAbstract {
  template<typename Model>
  ActionDataLQR(Model *const model) : ActionDataAbstract(model) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx = model->Fx;
    Fu = model->Fu;
    Lxx = model->Lxx;
    Luu = model->Luu;
    Lxu = model->Lxu;
  }
  ~ActionDataLQR() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_

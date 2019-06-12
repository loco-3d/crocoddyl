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

//TODO: ActionModelLQR ActionDataLQR
//TODO: DifferentialActionModelLQR DifferentialActionDataLQR

namespace crocoddyl {

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
  ~ActionDataLQR() { }
};

class ActionModelLQR : public ActionModelAbstract {
 public:
  ActionModelLQR(const int& nx, const int& nu, bool driftFree=true) : driftFree(driftFree),
      ActionModelAbstract(new StateVector(nx), nu) {
    //TODO substitute by random (vectors) and random-orthogonal (matrices)
    Fx = Eigen::MatrixXd::Identity(nx, nx);
    Fu = Eigen::MatrixXd::Identity(nx, nu);
    f0 = Eigen::VectorXd::Ones(nx);
    Lxx = Eigen::MatrixXd::Identity(nx, nx);
    Lxu = Eigen::MatrixXd::Identity(nx, nu);
    Luu = Eigen::MatrixXd::Identity(nu, nu);
    lx = Eigen::VectorXd::Ones(nx);
    lu = Eigen::VectorXd::Ones(nu);
  }
  ~ActionModelLQR() {}

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override {
    if (driftFree) {
      data->xnext = Fx * x + Fu * u;
    } else {
      data->xnext = Fx * x + Fu * u + f0;
    }
    data->cost = 0.5 * x.dot(Lxx * x) + 0.5 * u.dot(Luu * u) + x.dot(Lxu * u) + lx.dot(x) + lu.dot(u);
  }

  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc=true) override {
    if (recalc) {
      calc(data, x, u);
    }
    data->Lx = lx + Lxx * x + Lxu * u;
    data->Lu = lu + Lxu.transpose() * x + Luu * u;
    data->Fx = Fx;
    data->Fu = Fu;
    data->Lxx = Lxx;
    data->Lxu = Lxu;
    data->Luu = Luu;
  }

  std::shared_ptr<ActionDataAbstract> createData() override {
    return std::make_shared<ActionDataLQR>(this);
  }

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

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_

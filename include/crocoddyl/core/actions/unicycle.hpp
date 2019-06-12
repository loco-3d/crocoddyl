///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
#define CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

#include <crocoddyl/core/action-base.hpp>
//TODO: ActionModelUnicycleVar

namespace crocoddyl {

struct DataUnicycle : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename Model>
  DataUnicycle(Model *const model) : ActionDataAbstract(model) {
    const int& ncost = model->get_ncost();
    costResiduals = Eigen::VectorXd::Zero(ncost);
  }
  ~DataUnicycle() { }
  static const int ncost = 5;

  Eigen::Matrix<double, ncost, 1> costResiduals;
};

class ActionModelUnicycle : public ActionModelAbstract {
 public:
  ActionModelUnicycle(StateAbstract *const state) : ActionModelAbstract(state, 2), ncost(5), dt(0.1) {
    costWeights << 10., 1.;
  }
  ~ActionModelUnicycle() { }

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override {
    DataUnicycle* d = static_cast<DataUnicycle*>(data.get());
    const double& c = std::cos(x[2]);
    const double& s = std::sin(x[2]);
    d->xnext << x[0] + c * u[0] * dt,
                x[1] + s * u[0] * dt,
                x[2] + u[1] * dt;
    d->costResiduals.head<3>() = costWeights[0] * x;
    d->costResiduals.tail<2>() = costWeights[1] * u;
    d->cost = 0.5 * d->costResiduals.transpose() * d->costResiduals;
  }

  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc=true) override {
    if (recalc) {
      calc(data, x, u);
    }
    DataUnicycle* d = static_cast<DataUnicycle*>(data.get());

    // Cost derivatives
    const double& w_x = costWeights[0] * costWeights[0];
    const double& w_u = costWeights[1] * costWeights[1];
    d->Lx = x.cwiseProduct(Eigen::VectorXd::Constant(get_nx(), w_x));
    d->Lu = u.cwiseProduct(Eigen::VectorXd::Constant(get_nu(), w_u));
    d->Lxx.diagonal() << w_x, w_x, w_x;
    d->Luu.diagonal() << w_u, w_u;

    // Dynamic derivatives
    const double& c = std::cos(x[2]);
    const double& s = std::sin(x[2]);
    d->Fx << 1., 0., -s * u[0] * dt,
             0., 1., c * u[0] * dt,
             0., 0., 1.;
    d->Fu << c * dt, 0.,
             s * dt, 0.,
             0., dt;
  }
  std::shared_ptr<ActionDataAbstract> createData() override {
    return std::make_shared<DataUnicycle>(this);
  }

  int get_ncost() const {return ncost;}

 private:
  Eigen::Matrix<double, 2, 1> costWeights;
  int ncost;
  double dt;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

class ActionModelLQR : public ActionModelAbstract {
 public:
  ActionModelLQR(const std::size_t& nx, const std::size_t& nu, bool drift_free = true);
  ~ActionModelLQR();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<ActionDataAbstract> createData();

  Eigen::MatrixXd Fx_;
  Eigen::MatrixXd Fu_;
  Eigen::VectorXd f0_;
  Eigen::MatrixXd Lxx_;
  Eigen::MatrixXd Lxu_;
  Eigen::MatrixXd Luu_;
  Eigen::VectorXd lx_;
  Eigen::VectorXd lu_;

 private:
  bool drift_free_;
};

struct ActionDataLQR : public ActionDataAbstract {
  template <typename Model>
  explicit ActionDataLQR(Model* const model) : ActionDataAbstract(model) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx = model->Fx_;
    Fu = model->Fu_;
    Lxx = model->Lxx_;
    Luu = model->Luu_;
    Lxu = model->Lxu_;
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_

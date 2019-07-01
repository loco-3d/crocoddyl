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
#include <crocoddyl/core/states/state-euclidean.hpp>
// TODO: ActionModelUnicycleVar

namespace crocoddyl {

class ActionModelUnicycle : public ActionModelAbstract {
 public:
  ActionModelUnicycle();
  ~ActionModelUnicycle();

  void calc(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override;
  void calcDiff(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) override;
  std::shared_ptr<ActionDataAbstract> createData() override;

 private:
  Eigen::Matrix<double, 2, 1> cost_weights_;
  double dt_;
};

struct ActionDataUnicycle : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ActionDataUnicycle(Model* const model) : ActionDataAbstract(model) {}
  ~ActionDataUnicycle() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

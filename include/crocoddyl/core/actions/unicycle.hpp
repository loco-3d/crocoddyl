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

class ActionModelUnicycle : public ActionModelAbstract {
 public:
  ActionModelUnicycle(StateAbstract *const state);
  ~ActionModelUnicycle();

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override;
  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc=true) override;
  std::shared_ptr<ActionDataAbstract> createData() override;

  unsigned int get_ncost() const;

 private:
  Eigen::Matrix<double, 2, 1> costWeights;
  unsigned int ncost;
  double dt;
};

struct DataUnicycle : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename Model>
  DataUnicycle(Model *const model) : ActionDataAbstract(model) {
    const unsigned int& ncost = model->get_ncost();
    costResiduals = Eigen::VectorXd::Zero(ncost);
  }
  ~DataUnicycle() {}
  static const unsigned int ncost = 5;

  Eigen::Matrix<double, ncost, 1> costResiduals;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

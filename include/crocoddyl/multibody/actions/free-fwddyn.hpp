///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FREE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_FREE_FWDDYN_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class DifferentialActionModelFreeFwdDynamics : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelFreeFwdDynamics(StateMultibody* const state, CostModelSum* const costs);
  ~DifferentialActionModelFreeFwdDynamics();

  void calc(boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  CostModelSum* get_costs() const;
  pinocchio::Model* get_pinocchio() const;

 private:
  CostModelSum* costs_;
  pinocchio::Model* pinocchio_;
  bool force_aba_;
};

struct DifferentialActionDataFreeFwdDynamics : public DifferentialActionDataAbstract {
  template <typename Model>
  DifferentialActionDataFreeFwdDynamics(Model* const model)
      : DifferentialActionDataAbstract(model), pinocchio(pinocchio::Data(*model->get_pinocchio())), ddq_dq(model->get_nv(), model->get_nv()), ddq_dv(model->get_nv(), model->get_nv()), ddq_dtau(model->get_nv(), model->get_nv()) {
    costs = model->get_costs()->createData(&pinocchio);
    ddq_dq.fill(0);
    ddq_dv.fill(0);
    ddq_dtau.fill(0);
  }

  pinocchio::Data pinocchio;
  boost::shared_ptr<CostDataAbstract> costs;
  Eigen::MatrixXd ddq_dq;
  Eigen::MatrixXd ddq_dv;
  pinocchio::Data::RowMatrixXs ddq_dtau;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FREE_FWDDYN_HPP_

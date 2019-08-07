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

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  pinocchio::Model* get_pinocchio() const;
  CostModelSum* get_costs() const;
  const Eigen::VectorXd& get_armature() const;
  void set_armature(const Eigen::VectorXd& armature);

 private:
  CostModelSum* costs_;
  pinocchio::Model* pinocchio_;
  bool force_aba_;
  Eigen::VectorXd armature_;
};

struct DifferentialActionDataFreeFwdDynamics : public DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  DifferentialActionDataFreeFwdDynamics(Model* const model)
      : DifferentialActionDataAbstract(model),
        pinocchio(pinocchio::Data(*model->get_pinocchio())),
        Minv(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    costs = model->get_costs()->createData(&pinocchio);
    shareCostMemory(costs);
    Minv.fill(0);
  }

  pinocchio::Data pinocchio;
  boost::shared_ptr<CostDataAbstract> costs;
  Eigen::MatrixXd Minv;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FREE_FWDDYN_HPP_

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class DifferentialActionModelFreeFwdDynamics : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelFreeFwdDynamics(StateMultibody& state, CostModelSum& costs);
  ~DifferentialActionModelFreeFwdDynamics();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  CostModelSum& get_costs() const;
  pinocchio::Model& get_pinocchio() const;
  const Eigen::VectorXd& get_armature() const;
  void set_armature(const Eigen::VectorXd& armature);

 private:
  CostModelSum& costs_;
  pinocchio::Model& pinocchio_;
  bool with_armature_;
  Eigen::VectorXd armature_;
};

struct DifferentialActionDataFreeFwdDynamics : public DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit DifferentialActionDataFreeFwdDynamics(Model* const model)
      : DifferentialActionDataAbstract(model),
        pinocchio(pinocchio::Data(model->get_pinocchio())),
        q(model->get_state().get_nq()),
        v(model->get_state().get_nv()),
        Minv(model->get_state().get_nv(), model->get_state().get_nv()),
        u_drift(model->get_nu()) {
    costs = model->get_costs().createData(&pinocchio);
    costs->shareMemory(this);
    q.fill(0);
    v.fill(0);
    Minv.fill(0);
    u_drift.fill(0);
  }

  pinocchio::Data pinocchio;
  boost::shared_ptr<CostDataSum> costs;
  Eigen::VectorXd q;
  Eigen::VectorXd v;
  Eigen::MatrixXd Minv;
  Eigen::VectorXd u_drift;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_

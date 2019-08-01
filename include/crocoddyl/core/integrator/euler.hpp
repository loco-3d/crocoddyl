///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

class IntegratedActionModelEuler : public ActionModelAbstract {
 public:
  IntegratedActionModelEuler(DifferentialActionModelAbstract* const model, const double& time_step = 1e-3,
                             const bool& with_cost_residual = true);
  ~IntegratedActionModelEuler();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<ActionDataAbstract> createData();

  DifferentialActionModelAbstract* get_differential() const;
  const double& get_dt() const;
  void set_dt(double dt);

 private:
  DifferentialActionModelAbstract* differential_;
  double time_step_;
  double time_step2_;
  bool with_cost_residual_;
};

struct IntegratedActionDataEuler : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  IntegratedActionDataEuler(Model* const model) : ActionDataAbstract(model) {
    differential = model->get_differential()->createData();
    const unsigned int& ndx = model->get_ndx();
    const unsigned int& nu = model->get_nu();
    dx = Eigen::VectorXd::Zero(ndx);
    ddx_dx = Eigen::MatrixXd::Zero(ndx, ndx);
    ddx_du = Eigen::MatrixXd::Zero(ndx, nu);
    dxnext_dx = Eigen::MatrixXd::Zero(ndx, ndx);
    dxnext_ddx = Eigen::MatrixXd::Zero(ndx, ndx);
  }
  ~IntegratedActionDataEuler() {}

  boost::shared_ptr<DifferentialActionDataAbstract> differential;
  Eigen::VectorXd dx;
  Eigen::MatrixXd ddx_dx;
  Eigen::MatrixXd ddx_du;
  Eigen::MatrixXd dxnext_dx;
  Eigen::MatrixXd dxnext_ddx;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

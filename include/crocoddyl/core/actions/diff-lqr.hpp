///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

struct DifferentialActionDataLQR;  // forward declaration

class DifferentialActionModelLQR : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelLQR(unsigned int const& nq, unsigned int const& nu, bool drift_free = true);
  ~DifferentialActionModelLQR();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  Eigen::MatrixXd Fq_;
  Eigen::MatrixXd Fv_;
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

struct DifferentialActionDataLQR : public DifferentialActionDataAbstract {
  template <typename Model>
  explicit DifferentialActionDataLQR(Model* const model)
      : DifferentialActionDataAbstract(model), q(model->get_state().get_nq()), v(model->get_state().get_nv()) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx.leftCols(model->get_state().get_nq()) = model->Fq_;
    Fx.rightCols(model->get_state().get_nv()) = model->Fv_;
    Fu = model->Fu_;
    Lxx = model->Lxx_;
    Luu = model->Luu_;
    Lxu = model->Lxu_;

    q.fill(0);
    v.fill(0);
  }

  Eigen::VectorXd q;
  Eigen::VectorXd v;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

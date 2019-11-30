///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

struct DifferentialActionDataLQR;  // forward declaration

class DifferentialActionModelLQR : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelLQR(const std::size_t& nq, const std::size_t& nu, bool drift_free = true);
  ~DifferentialActionModelLQR();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  const Eigen::MatrixXd& get_Fq() const;
  const Eigen::MatrixXd& get_Fv() const;
  const Eigen::MatrixXd& get_Fu() const;
  const Eigen::VectorXd& get_f0() const;
  const Eigen::VectorXd& get_lx() const;
  const Eigen::VectorXd& get_lu() const;
  const Eigen::MatrixXd& get_Lxx() const;
  const Eigen::MatrixXd& get_Lxu() const;
  const Eigen::MatrixXd& get_Luu() const;

  void set_Fq(const Eigen::MatrixXd& Fq);
  void set_Fv(const Eigen::MatrixXd& Fv);
  void set_Fu(const Eigen::MatrixXd& Fu);
  void set_f0(const Eigen::VectorXd& f0);
  void set_lx(const Eigen::VectorXd& lx);
  void set_lu(const Eigen::VectorXd& lu);
  void set_Lxx(const Eigen::MatrixXd& Lxx);
  void set_Lxu(const Eigen::MatrixXd& Lxu);
  void set_Luu(const Eigen::MatrixXd& Luu);

 private:
  bool drift_free_;
  Eigen::MatrixXd Fq_;
  Eigen::MatrixXd Fv_;
  Eigen::MatrixXd Fu_;
  Eigen::VectorXd f0_;
  Eigen::MatrixXd Lxx_;
  Eigen::MatrixXd Lxu_;
  Eigen::MatrixXd Luu_;
  Eigen::VectorXd lx_;
  Eigen::VectorXd lu_;
};

struct DifferentialActionDataLQR : public DifferentialActionDataAbstract {
  template <typename Model>
  explicit DifferentialActionDataLQR(Model* const model) : DifferentialActionDataAbstract(model) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx.leftCols(model->get_state()->get_nq()) = model->get_Fq();
    Fx.rightCols(model->get_state()->get_nv()) = model->get_Fv();
    Fu = model->get_Fu();
    Lxx = model->get_Lxx();
    Luu = model->get_Luu();
    Lxu = model->get_Lxu();
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

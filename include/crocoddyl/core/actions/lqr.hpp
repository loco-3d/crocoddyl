///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include <stdexcept>

namespace crocoddyl {

class ActionModelLQR : public ActionModelAbstract {
 public:
  ActionModelLQR(const std::size_t& nx, const std::size_t& nu, bool drift_free = true);
  ~ActionModelLQR();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<ActionDataAbstract> createData();

  const Eigen::MatrixXd& get_Fx() const;
  const Eigen::MatrixXd& get_Fu() const;
  const Eigen::VectorXd& get_f0() const;
  const Eigen::VectorXd& get_lx() const;
  const Eigen::VectorXd& get_lu() const;
  const Eigen::MatrixXd& get_Lxx() const;
  const Eigen::MatrixXd& get_Lxu() const;
  const Eigen::MatrixXd& get_Luu() const;

  void set_Fx(const Eigen::MatrixXd& Fx);
  void set_Fu(const Eigen::MatrixXd& Fu);
  void set_f0(const Eigen::VectorXd& f0);
  void set_lx(const Eigen::VectorXd& lx);
  void set_lu(const Eigen::VectorXd& lu);
  void set_Lxx(const Eigen::MatrixXd& Lxx);
  void set_Lxu(const Eigen::MatrixXd& Lxu);
  void set_Luu(const Eigen::MatrixXd& Luu);

 private:
  bool drift_free_;
  Eigen::MatrixXd Fx_;
  Eigen::MatrixXd Fu_;
  Eigen::VectorXd f0_;
  Eigen::MatrixXd Lxx_;
  Eigen::MatrixXd Lxu_;
  Eigen::MatrixXd Luu_;
  Eigen::VectorXd lx_;
  Eigen::VectorXd lu_;
};

struct ActionDataLQR : public ActionDataAbstract {
  template<template<typename Scalar> class Model>
  explicit ActionDataLQR(Model<Scalar>* const model) : ActionDataAbstract(model) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx = model->get_Fx();
    Fu = model->get_Fu();
    Lxx = model->get_Lxx();
    Luu = model->get_Luu();
    Lxu = model->get_Lxu();
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_

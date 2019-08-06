///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTION_BASE_HPP_
#define CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace crocoddyl {

struct ActionDataAbstract;  // forward declaration

class ActionModelAbstract {
 public:
  ActionModelAbstract(StateAbstract* const state, const unsigned int& nu, const unsigned int& nr = 1);
  virtual ~ActionModelAbstract();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) = 0;
  virtual boost::shared_ptr<ActionDataAbstract> createData() = 0;

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  const unsigned int& get_nx() const;
  const unsigned int& get_ndx() const;
  const unsigned int& get_nu() const;
  const unsigned int& get_nr() const;
  StateAbstract* get_state() const;

 protected:
  unsigned int nx_;
  unsigned int ndx_;
  unsigned int nu_;
  unsigned int nr_;
  StateAbstract* state_;
  Eigen::VectorXd unone_;

#ifdef PYTHON_BINDINGS
 public:
  void calc_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u) {
    calc(data, x, u);
  }
  void calc_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calc(data, x, unone_);
  }

  void calcDiff_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u, const bool& recalc) {
    calcDiff(data, x, u, recalc);
  }
  void calcDiff_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, const bool& recalc) {
    calcDiff(data, x, unone_, recalc);
  }
#endif
};

struct ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ActionDataAbstract(Model* const model) : cost(0.) {
    const int& nx = model->get_nx();
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();
    const int& nr = model->get_nr();
    xnext = Eigen::VectorXd::Zero(nx);
    Fx = Eigen::MatrixXd::Zero(ndx, ndx);
    Fu = Eigen::MatrixXd::Zero(ndx, nu);
    Lx = Eigen::VectorXd::Zero(ndx);
    Lu = Eigen::VectorXd::Zero(nu);
    Lxx = Eigen::MatrixXd::Zero(ndx, ndx);
    Lxu = Eigen::MatrixXd::Zero(ndx, nu);
    Luu = Eigen::MatrixXd::Zero(nu, nu);
    r = Eigen::VectorXd::Zero(nr);
    Rx = Eigen::MatrixXd::Zero(nr, ndx);
    Ru = Eigen::MatrixXd::Zero(nr, nu);
  }

  const double& get_cost() const { return cost; }
  const Eigen::VectorXd& get_xnext() const { return xnext; }
  const Eigen::VectorXd& get_Lx() const { return Lx; }
  const Eigen::VectorXd& get_Lu() const { return Lu; }
  const Eigen::MatrixXd& get_Lxx() const { return Lxx; }
  const Eigen::MatrixXd& get_Lxu() const { return Lxu; }
  const Eigen::MatrixXd& get_Luu() const { return Luu; }
  const Eigen::MatrixXd& get_Fx() const { return Fx; }
  const Eigen::MatrixXd& get_Fu() const { return Fu; }
  const Eigen::VectorXd& get_r() const { return r; }
  const Eigen::MatrixXd& get_Rx() const { return Rx; }
  const Eigen::MatrixXd& get_Ru() const { return Ru; }

  double cost;
  Eigen::VectorXd xnext;
  Eigen::MatrixXd Fx;
  Eigen::MatrixXd Fu;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
  Eigen::VectorXd r;
  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_

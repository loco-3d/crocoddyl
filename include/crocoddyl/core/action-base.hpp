///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTION_BASE_HPP_
#define CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace crocoddyl {

struct ActionDataAbstract;  // forward declaration

class ActionModelAbstract {
 public:
  ActionModelAbstract(StateAbstract& state, unsigned int const& nu, unsigned int const& nr = 0);
  virtual ~ActionModelAbstract();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) = 0;
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x, unsigned int const& maxiter = 100,
                   const double& tol = 1e-9);

  unsigned int const& get_nu() const;
  unsigned int const& get_nr() const;
  StateAbstract& get_state() const;

  const Eigen::VectorXd& get_u_lb() const;
  const Eigen::VectorXd& get_u_ub() const;
  bool const& get_has_control_limits() const;

  void set_u_lb(const Eigen::Ref<const Eigen::VectorXd>& u_in);
  void set_u_ub(const Eigen::Ref<const Eigen::VectorXd>& u_in);

 protected:
  unsigned int nu_;          //!< Control dimension
  unsigned int nr_;          //!< Dimension of the cost residual
  StateAbstract& state_;     //!< Model of the state
  Eigen::VectorXd unone_;    //!< Neutral state
  Eigen::VectorXd u_lb_;     //!< Lower control limits
  Eigen::VectorXd u_ub_;     //!< Upper control limits
  bool has_control_limits_;  //!< Indicates whether any of the control limits is finite

  void update_has_control_limits();

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u = Eigen::VectorXd()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
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

  Eigen::VectorXd quasiStatic_wrap(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x,
                                   unsigned int const& maxiter = 100, const double& tol = 1e-9) {
    Eigen::VectorXd u(nu_);
    u.setZero();
    quasiStatic(data, u, x, maxiter, tol);
    return u;
  }

#endif
};

struct ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit ActionDataAbstract(Model* const model)
      : cost(0.),
        xnext(model->get_state().get_nx()),
        r(model->get_nr()),
        Fx(model->get_state().get_ndx(), model->get_state().get_ndx()),
        Fu(model->get_state().get_ndx(), model->get_nu()),
        Lx(model->get_state().get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state().get_ndx(), model->get_state().get_ndx()),
        Lxu(model->get_state().get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()) {
    xnext.setZero();
    r.setZero();
    Fx.setZero();
    Fu.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }

  double cost;
  Eigen::VectorXd xnext;
  Eigen::VectorXd r;
  Eigen::MatrixXd Fx;
  Eigen::MatrixXd Fu;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_

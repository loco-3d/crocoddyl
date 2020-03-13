///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

/**
 * @brief This class DifferentialActionModelAbstract represents a first-order
 * ODE, i.e.
 * \f[
 * \mathbf{\dot{v}} = \mathbf{f}(\mathbf{q}, \mathbf{v}, \boldsymbol{\tau})
 * \f]
 * where \f$ xout = \mathbf{\dot{v}} \f$ and represents the  acceleration of the
 * system. Note that Jacobians Fx and Fu in the
 * DifferentialActionDataAbstract are in \f$ \mathbb{R}^{nv\times ndx} \f$ and
 * \f$ \mathbb{R}^{nv\times nu} \f$, respectively.
 *
 * Then we use the acceleration to integrate the system, and as consequence we
 * obtain:
 * \f[
 * \mathbf{\dot{x}} = (\mathbf{v}, \mathbf{\dot{v}}) = \mathbf{f}(\mathbf{x},\mathbf{u})
 * \f]
 * where this \f$ f \f$ function is different to the other one.
 * So \f$ xout \f$ is interpreted here as \f$ vdout \f$ or \f$ aout \f$.
 */

template <typename _Scalar>
class DifferentialActionModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t& nu,
                                     const std::size_t& nr = 0);
  virtual ~DifferentialActionModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) = 0;
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  const std::size_t& get_nu() const;
  const std::size_t& get_nr() const;
  const boost::shared_ptr<StateAbstract>& get_state() const;

  const VectorXs& get_u_lb() const;
  const VectorXs& get_u_ub() const;
  bool const& get_has_control_limits() const;

  void set_u_lb(const VectorXs& u_lb);
  void set_u_ub(const VectorXs& u_ub);

 protected:
  std::size_t nu_;                          //!< Control dimension
  std::size_t nr_;                          //!< Dimension of the cost residual
  boost::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;                          //!< Neutral state
  VectorXs u_lb_;                           //!< Lower control limits
  VectorXs u_ub_;                           //!< Upper control limits
  bool has_control_limits_;                 //!< Indicates whether any of the control limits is finite

  void update_has_control_limits();

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const VectorXs& x,
                 const VectorXs& u = VectorXs()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const VectorXs& x,
                     const VectorXs& u) {
    calcDiff(data, x, u);
  }
  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const VectorXs& x) {
    calcDiff(data, x, unone_);
  }

#endif
};

template <typename _Scalar>
struct DifferentialActionDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataAbstractTpl(Model<Scalar>* const model)
      : cost(0.),
        xout(model->get_state()->get_nv()),
        Fx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        Fu(model->get_state()->get_nv(), model->get_nu()),
        r(model->get_nr()),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()) {
    xout.setZero();
    r.setZero();
    Fx.setZero();
    Fu.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }
  virtual ~DifferentialActionDataAbstractTpl() {}

  Scalar cost;
  VectorXs xout;
  MatrixXs Fx;
  MatrixXs Fu;
  VectorXs r;
  VectorXs Lx;
  VectorXs Lu;
  MatrixXs Lxx;
  MatrixXs Lxu;
  MatrixXs Luu;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/diff-action-base.hxx"
#endif  // CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

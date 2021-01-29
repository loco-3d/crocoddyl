///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include <stdexcept>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActionModelLQRTpl : public ActionModelAbstractTpl<_Scalar> {
public:
  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataLQRTpl<Scalar> Data;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActionModelLQRTpl(const std::size_t &nx, const std::size_t &nu,
                    bool drift_free = true);
  virtual ~ActionModelLQRTpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

  const MatrixXs &get_Fx() const;
  const MatrixXs &get_Fu() const;
  const VectorXs &get_f0() const;
  const VectorXs &get_lx() const;
  const VectorXs &get_lu() const;
  const MatrixXs &get_Lxx() const;
  const MatrixXs &get_Lxu() const;
  const MatrixXs &get_Luu() const;

  void set_Fx(const MatrixXs &Fx);
  void set_Fu(const MatrixXs &Fu);
  void set_f0(const VectorXs &f0);
  void set_lx(const VectorXs &lx);
  void set_lu(const VectorXs &lu);
  void set_Lxx(const MatrixXs &Lxx);
  void set_Lxu(const MatrixXs &Lxu);
  void set_Luu(const MatrixXs &Luu);

protected:
  using Base::has_control_limits_; //!< Indicates whether any of the control
                                   //!< limits
  using Base::nr_;                 //!< Dimension of the cost residual
  using Base::nu_;                 //!< Control dimension
  using Base::state_;              //!< Model of the state
  using Base::u_lb_;               //!< Lower control limits
  using Base::u_ub_;               //!< Upper control limits
  using Base::unone_;              //!< Neutral state

private:
  bool drift_free_;
  MatrixXs Fx_;
  MatrixXs Fu_;
  VectorXs f0_;
  MatrixXs Lxx_;
  MatrixXs Lxu_;
  MatrixXs Luu_;
  VectorXs lx_;
  VectorXs lu_;
};

template <typename _Scalar>
struct ActionDataLQRTpl : public ActionDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;

  template <template <typename Scalar> class Model>
  explicit ActionDataLQRTpl(Model<Scalar> *const model) : Base(model) {
    // Setting the linear model and quadratic cost here because they are
    // constant
    Fx = model->get_Fx();
    Fu = model->get_Fu();
    Lxx = model->get_Lxx();
    Luu = model->get_Luu();
    Lxu = model->get_Lxu();
  }

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;
};

} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/actions/lqr.hxx"

#endif // CROCODDYL_CORE_ACTIONS_LQR_HPP_

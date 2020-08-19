///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK4_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"

template <typename _Scalar>
class IntegratedActionModelRK4Tpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedActionModelRK4Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              const Scalar& time_step = Scalar(1e-3), const bool& with_cost_residual = true);
  virtual ~IntegratedActionModelRK4Tpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t& maxiter = 100,
                           const Scalar& tol = Scalar(1e-9));

  const boost::shared_ptr<DifferentialActionModelAbstract>& get_differential() const;
  const Scalar& get_dt() const;

  void set_dt(const Scalar& dt);
  void set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

 private:
  boost::shared_ptr<DifferentialActionModelAbstract> differential_;
  Scalar time_step_;
  std::vector<Scalar> rk4_c_;
  bool with_cost_residual_;
  bool enable_integration_;
};

template <typename _Scalar>
struct IntegratedActionDataEulerTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataEulerTpl(Model<Scalar>* const model) : Base(model) {
    differential = std::vector<boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>>(
        4, model->get_differential()->createData());
    const std::size_t& ndx = model->get_state()->get_ndx();
    const std::size_t& nx = model->get_state()->get_nx();

    dx_rk4 = VectorXs::Zero(ndx);
    integral = VectorXs::Zero(4);
    ki = std::vector<VectorXs>(4, VectorXs::Zero(ndx));
    y = std::vector<VectorXs>(4, VectorXs::Zero(nx));
    dx = std::vector<VectorXs>(4, VectorXs::Zero(nx));
    
  }
  ~IntegratedActionDataEulerTpl() {}

  VectorXs dx_rk4;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>> differential;
  std::vector<Scalar> integral;
  std::vector<VectorXs> ki;
  std::vector<VectorXs> y;
  std::vector<VectorXs> dx;

  std::vector<MatrixXs> dki_dx;
  std::vector<MatrixXs> dy_dx;

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

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/integrator/euler.hxx"

#endif  // CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

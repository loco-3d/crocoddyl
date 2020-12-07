///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK4_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class IntegratedActionModelRK4Tpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataRK4Tpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedActionModelRK4Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              Scalar time_step = Scalar(1e-3), bool with_cost_residual = true);
  virtual ~IntegratedActionModelRK4Tpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, std::size_t maxiter = 100,
                           Scalar tol = Scalar(1e-9));

  const boost::shared_ptr<DifferentialActionModelAbstract>& get_differential() const;
  Scalar get_dt() const;

  void set_dt(Scalar dt);
  void set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits are active
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
struct IntegratedActionDataRK4Tpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataRK4Tpl(Model<Scalar>* const model) : Base(model) {
    std::size_t ndx = model->get_state()->get_ndx();
    std::size_t nx = model->get_state()->get_nx();
    std::size_t nv = model->get_state()->get_nv();
    std::size_t nu = model->get_nu();

    for (std::size_t i = 0; i < 4; ++i) {
      differential.push_back(
          boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >(model->get_differential()->createData()));
    }

    dx = VectorXs::Zero(ndx);
    integral = std::vector<Scalar>(4, Scalar(0.));

    ki = std::vector<VectorXs>(4, VectorXs::Zero(ndx));
    y = std::vector<VectorXs>(4, VectorXs::Zero(nx));
    dx_rk4 = std::vector<VectorXs>(4, VectorXs::Zero(ndx));

    dki_dx = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, ndx));
    dki_du = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, nu));
    dyi_dx = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, ndx));
    dyi_du = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, nu));
    dki_dy = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, ndx));

    dli_dx = std::vector<VectorXs>(4, VectorXs::Zero(ndx));
    dli_du = std::vector<VectorXs>(4, VectorXs::Zero(nu));
    ddli_ddx = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, ndx));
    ddli_ddu = std::vector<MatrixXs>(4, MatrixXs::Zero(nu, nu));
    ddli_dxdu = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, nu));
    Luu_partialx = std::vector<MatrixXs>(4, MatrixXs::Zero(nu, nu));
    Lxx_partialx = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, ndx));
    Lxx_partialu = std::vector<MatrixXs>(4, MatrixXs::Zero(ndx, nu));

    dyi_dx[0].diagonal().array() = (Scalar)1;
    for (std::size_t i = 0; i < 4; ++i) {
      dki_dy[i].topRightCorner(nv, nv).diagonal().array() = (Scalar)1;
    }
  }
  virtual ~IntegratedActionDataRK4Tpl() {}

  VectorXs dx;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > > differential;
  std::vector<Scalar> integral;
  std::vector<VectorXs> ki;
  std::vector<VectorXs> y;
  std::vector<VectorXs> dx_rk4;

  std::vector<MatrixXs> dki_dx;
  std::vector<MatrixXs> dki_du;
  std::vector<MatrixXs> dyi_dx;
  std::vector<MatrixXs> dyi_du;
  std::vector<MatrixXs> dki_dy;

  std::vector<VectorXs> dli_dx;
  std::vector<VectorXs> dli_du;
  std::vector<MatrixXs> ddli_ddx;
  std::vector<MatrixXs> ddli_ddu;
  std::vector<MatrixXs> ddli_dxdu;
  std::vector<MatrixXs> Luu_partialx;
  std::vector<MatrixXs> Lxx_partialx;
  std::vector<MatrixXs> Lxx_partialu;

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
#include "crocoddyl/core/integrator/rk4.hxx"

#endif  // CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

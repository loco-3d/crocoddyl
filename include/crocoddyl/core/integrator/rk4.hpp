///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK4_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class IntegratedActionModelRK4Tpl : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataRK4Tpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> ControlParametrizationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedActionModelRK4Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  IntegratedActionModelRK4Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              boost::shared_ptr<ControlParametrizationModelAbstract> control,
                              const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  virtual ~IntegratedActionModelRK4Tpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Print relevant information of the Runge-Kutta 4 integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::control_;             //!< Control parametrization
  using Base::differential_;        //!< Differential action model
  using Base::enable_integration_;  //!< False for the terminal horizon node, where integration is not needed
  using Base::nu_;                  //!< Dimension of the control
  using Base::state_;               //!< Model of the state
  using Base::time_step2_;          //!< Square of the time step used for integration
  using Base::time_step_;           //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual is used

 private:
  std::vector<Scalar> rk4_c_;
};

template <typename _Scalar>
struct IntegratedActionDataRK4Tpl : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataRK4Tpl(Model<Scalar>* const model)
      : Base(model),
        integral(4, Scalar(0.)),
        dx(model->get_state()->get_ndx()),
        ki(4, VectorXs::Zero(model->get_state()->get_ndx())),
        y(4, VectorXs::Zero(model->get_state()->get_nx())),
        ws(4, VectorXs::Zero(model->get_control()->get_nw())),
        dx_rk4(4, VectorXs::Zero(model->get_state()->get_ndx())),
        dki_dx(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        dki_dw(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_control()->get_nw())),
        dki_du(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dki_dy(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        dfi_du(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dyi_dx(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        dyi_du(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dli_dx(4, VectorXs::Zero(model->get_state()->get_ndx())),
        dli_du(4, VectorXs::Zero(model->get_nu())),
        ddli_ddx(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        ddli_ddw(4, MatrixXs::Zero(model->get_control()->get_nw(), model->get_control()->get_nw())),
        ddli_dwdu(4, MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu())),
        ddli_ddu(4, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        ddli_dxdw(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_control()->get_nw())),
        ddli_dxdu(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        Luu_partialx(4, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        Lxu_i(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        Lxx_partialx(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        Lxx_partialu(4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())) {
    dx.setZero();

    for (std::size_t i = 0; i < 4; ++i) {
      differential.push_back(
          boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >(model->get_differential()->createData()));
    }

    const std::size_t nv = model->get_state()->get_nv();
    dyi_dx[0].diagonal().array() = (Scalar)1;
    for (std::size_t i = 0; i < 4; ++i) {
      dki_dy[i].topRightCorner(nv, nv).diagonal().array() = (Scalar)1;
    }
  }
  virtual ~IntegratedActionDataRK4Tpl() {}

  std::vector<boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > > differential;
  std::vector<Scalar> integral;
  VectorXs dx;
  std::vector<VectorXs> ki;
  std::vector<VectorXs> y;
  std::vector<VectorXs> ws;
  std::vector<VectorXs> dx_rk4;

  std::vector<MatrixXs> dki_dx;
  std::vector<MatrixXs> dki_dw;
  std::vector<MatrixXs> dki_du;
  std::vector<MatrixXs> dki_dy;
  std::vector<MatrixXs> dfi_du;

  std::vector<MatrixXs> dyi_dx;
  std::vector<MatrixXs> dyi_du;

  std::vector<VectorXs> dli_dx;
  std::vector<VectorXs> dli_du;

  std::vector<MatrixXs> ddli_ddx;
  std::vector<MatrixXs> ddli_ddw;
  std::vector<MatrixXs> ddli_dwdu;
  std::vector<MatrixXs> ddli_ddu;
  std::vector<MatrixXs> ddli_dxdw;
  std::vector<MatrixXs> ddli_dxdu;

  std::vector<MatrixXs> Luu_partialx;
  std::vector<MatrixXs> Lxu_i;
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

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK3_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK3_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {

/**
 * @brief Semi-implicit RK3 integrator
 *
 * It applies a standard RK3 integration scheme to a differential (i.e., continuous time) action model.
 *
 * This standard RK4 scheme introduces also the possibility to parametrize the control trajectory inside an
 * integration step, for instance using polynomials. This requires introducing some notation to clarify the difference
 * between the control inputs of the differential model and the control inputs to the integrated model. We have decided
 * to use \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and \f$\mathbf{u}\f$ for the
 * control inputs of the integrated action model.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelRK3Tpl : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataRK3Tpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> ControlParametrizationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the RK3 integrator
   *
   * @param[in] model      Differential action model
   * @param[in] control    Control parametrization
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelRK3Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              boost::shared_ptr<ControlParametrizationModelAbstract> control,
                              const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);

  /**
   * @brief Initialize the RK3 integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the control parametrization.
   *
   * @param[in] model      Differential action model
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelRK3Tpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                              const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  virtual ~IntegratedActionModelRK3Tpl();

  /**
   * @brief Integrate the differential action model using RK3 scheme
   *
   * @param[in] data  Semi-implicit RK3 data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Integrate the total cost value for nodes that depends only on the state using RK3 scheme
   *
   * It computes the total cost and defines the next state as the current one. This function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Semi-implicit RK3 data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the partial derivatives of the RK3 integrator
   *
   * @param[in] data  Semi-implicit RK3 data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the partial derivatives of the cost
   *
   * It updates the derivatives of the cost function with respect to the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Semi-implicit RK3 data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the RK3 data
   *
   * @return the RK3 data
   */
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference posture as an equilibrium point, i.e.
   * for \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data    Semi-implicit RK3 data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Print relevant information of the RK3 integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::control_;             //!< Control parametrization
  using Base::differential_;        //!< Differential action model
  using Base::nu_;                  //!< Dimension of the control
  using Base::state_;               //!< Model of the state
  using Base::time_step_;           //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual is used

 private:
  std::array<Scalar, 3> rk3_c_;
};

template <typename _Scalar>
struct IntegratedActionDataRK3Tpl : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataRK3Tpl(Model<Scalar>* const model)
      : Base(model),
        integral(3, Scalar(0.)),
        dx(model->get_state()->get_ndx()),
        ki(3, VectorXs::Zero(model->get_state()->get_ndx())),
        y(3, VectorXs::Zero(model->get_state()->get_nx())),
        ws(3, VectorXs::Zero(model->get_control()->get_nw())),
        dx_rk3(3, VectorXs::Zero(model->get_state()->get_ndx())),
        dki_dx(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        dki_du(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dyi_dx(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        dyi_du(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dli_dx(3, VectorXs::Zero(model->get_state()->get_ndx())),
        dli_du(3, VectorXs::Zero(model->get_nu())),
        ddli_ddx(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        ddli_ddw(3, MatrixXs::Zero(model->get_control()->get_nw(), model->get_control()->get_nw())),
        ddli_ddu(3, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        ddli_dxdw(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_control()->get_nw())),
        ddli_dxdu(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        ddli_dwdu(3, MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu())),
        Luu_partialx(3, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        Lxu_i(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        Lxx_partialx(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_state()->get_ndx())),
        Lxx_partialu(3, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())) {
    dx.setZero();

    for (std::size_t i = 0; i < 3; ++i) {
      differential.push_back(
          boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >(model->get_differential()->createData()));
      control.push_back(boost::shared_ptr<ControlParametrizationDataAbstract>(model->get_control()->createData()));
    }

    const std::size_t nv = model->get_state()->get_nv();
    dyi_dx[0].diagonal().setOnes();
    dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
  }
  virtual ~IntegratedActionDataRK3Tpl() {}

  std::vector<boost::shared_ptr<DifferentialActionDataAbstract> > differential;  //!< List of differential model data
  std::vector<boost::shared_ptr<ControlParametrizationDataAbstract> >
      control;  //!< List of control parametrization data
  std::vector<Scalar> integral;
  VectorXs dx;               //!< State rate
  std::vector<VectorXs> ki;  //!< List of RK3 terms related to system dynamics
  std::vector<VectorXs> y;   //!< List of states where f is evaluated in the RK3 integration
  std::vector<VectorXs> ws;  //!< Control inputs evaluated in the RK4 integration
  std::vector<VectorXs> dx_rk3;

  std::vector<MatrixXs>
      dki_dx;  //!< List of partial derivatives of RK4 nodes with respect to the state of the RK3 integration. dki/dx
  std::vector<MatrixXs> dki_du;  //!< List of partial derivatives of RK4 nodes with respect to the control parameters
                                 //!< of the RK3 integration. dki/du

  std::vector<MatrixXs>
      dyi_dx;  //!< List of partial derivatives of RK4 dynamics with respect to the state of the RK3 integrator. dyi/dx
  std::vector<MatrixXs> dyi_du;  //!< List of partial derivatives of RK4 dynamics with respect to the control
                                 //!< parameters of the RK4 integrator. dyi/du

  std::vector<VectorXs>
      dli_dx;  //!< List of partial derivatives of the cost with respect to the state of the RK3 integration. dli_dx
  std::vector<VectorXs> dli_du;  //!< List of partial derivatives of the cost with respect to the control input of the
                                 //!< RK3 integration. dli_du

  std::vector<MatrixXs> ddli_ddx;  //!< List of second partial derivatives of the cost with respect to the state of the
                                   //!< RK3 integration. ddli_ddx
  std::vector<MatrixXs> ddli_ddw;  //!< List of second partial derivatives of the cost with respect to the control
                                   //!< parameters of the RK3 integration. ddli_ddw
  std::vector<MatrixXs> ddli_ddu;  //!< List of second partial derivatives of the cost with respect to the control
                                   //!< input of the RK3 integration. ddli_ddu
  std::vector<MatrixXs> ddli_dxdw;  //!< List of second partial derivatives of the cost with respect to the state and
                                    //!< control input of the RK3 integration. ddli_dxdw
  std::vector<MatrixXs> ddli_dxdu;  //!< List of second partial derivatives of the cost with respect to the state and
                                    //!< control parameters of the RK3 integration. ddli_dxdu
  std::vector<MatrixXs> ddli_dwdu;  //!< List of second partial derivatives of the cost with respect to the control
                                    //!< parameters and inputs control of the RK3 integration. ddli_dxdu

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
#include "crocoddyl/core/integrator/rk3.hxx"

#endif  // CROCODDYL_CORE_INTEGRATOR_RK3_HPP_

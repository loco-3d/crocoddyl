///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK4_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Standard RK4 integrator
 *
 * It applies a standard RK4 integration scheme to a differential (i.e.,
 * continuous time) action model.
 *
 * This standard RK4 scheme introduces also the possibility to parametrize the
 * control trajectory inside an integration step, for instance using
 * polynomials. This requires introducing some notation to clarify the
 * difference between the control inputs of the differential model and the
 * control inputs to the integrated model. We have decided to use
 * \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and
 * \f$\mathbf{u}\f$ for the control inputs of the integrated action model.
 *
 * \sa `IntegratedActionModelAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelRK4Tpl
    : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataRK4Tpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar>
      ControlParametrizationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the RK4 integrator
   *
   * @param[in] model      Differential action model
   * @param[in] control    Control parametrization
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  DEPRECATED("Use IntegratedActionModelRK",
             IntegratedActionModelRK4Tpl(
                 std::shared_ptr<DifferentialActionModelAbstract> model,
                 std::shared_ptr<ControlParametrizationModelAbstract> control,
                 const Scalar time_step = Scalar(1e-3),
                 const bool with_cost_residual = true);)

  /**
   * @brief Initialize the RK4 integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the
   * control parametrization.
   *
   * @param[in] model      Differential action model
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  DEPRECATED("Use IntegratedActionModelRK",
             IntegratedActionModelRK4Tpl(
                 std::shared_ptr<DifferentialActionModelAbstract> model,
                 const Scalar time_step = Scalar(1e-3),
                 const bool with_cost_residual = true);)
  virtual ~IntegratedActionModelRK4Tpl();

  /**
   * @brief Integrate the differential action model using RK4 scheme
   *
   * @param[in] data  RK4 integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Integrate the total cost value for nodes that depends only on the
   * state using RK4 scheme
   *
   * It computes the total cost and defines the next state as the current one.
   * This function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  RK4 integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the partial derivatives of the RK4 integrator
   *
   * @param[in] data  RK4 integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the partial derivatives of the cost
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  RK4 integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the RK4 integrator data
   *
   * @return the RK4 integrator data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const std::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference
   * posture as an equilibrium point, i.e. for
   * \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data    RK4 integrator data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Print relevant information of the RK4 integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::control_;       //!< Control parametrization
  using Base::differential_;  //!< Differential action model
  using Base::ng_;            //!< Number of inequality constraints
  using Base::nh_;            //!< Number of equality constraints
  using Base::nu_;            //!< Dimension of the control
  using Base::state_;         //!< Model of the state
  using Base::time_step2_;    //!< Square of the time step used for integration
  using Base::time_step_;     //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual
                                    //!< is used

 private:
  std::array<Scalar, 4> rk4_c_;
};

template <typename _Scalar>
struct IntegratedActionDataRK4Tpl
    : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
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
        dki_dx(4, MatrixXs::Zero(model->get_state()->get_ndx(),
                                 model->get_state()->get_ndx())),
        dki_du(4,
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dyi_dx(4, MatrixXs::Zero(model->get_state()->get_ndx(),
                                 model->get_state()->get_ndx())),
        dyi_du(4,
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dli_dx(4, VectorXs::Zero(model->get_state()->get_ndx())),
        dli_du(4, VectorXs::Zero(model->get_nu())),
        ddli_ddx(4, MatrixXs::Zero(model->get_state()->get_ndx(),
                                   model->get_state()->get_ndx())),
        ddli_ddw(4, MatrixXs::Zero(model->get_control()->get_nw(),
                                   model->get_control()->get_nw())),
        ddli_ddu(4, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        ddli_dxdw(4, MatrixXs::Zero(model->get_state()->get_ndx(),
                                    model->get_control()->get_nw())),
        ddli_dxdu(
            4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        ddli_dwdu(
            4, MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu())),
        Luu_partialx(4, MatrixXs::Zero(model->get_nu(), model->get_nu())),
        Lxu_i(4,
              MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        Lxx_partialx(4, MatrixXs::Zero(model->get_state()->get_ndx(),
                                       model->get_state()->get_ndx())),
        Lxx_partialu(
            4, MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())) {
    dx.setZero();

    for (std::size_t i = 0; i < 4; ++i) {
      differential.push_back(std::shared_ptr<DifferentialActionDataAbstract>(
          model->get_differential()->createData()));
      control.push_back(std::shared_ptr<ControlParametrizationDataAbstract>(
          model->get_control()->createData()));
    }

    const std::size_t nv = model->get_state()->get_nv();
    dyi_dx[0].diagonal().setOnes();
    dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
  }
  virtual ~IntegratedActionDataRK4Tpl() {}

  std::vector<std::shared_ptr<DifferentialActionDataAbstract> >
      differential;  //!< List of differential model data
  std::vector<std::shared_ptr<ControlParametrizationDataAbstract> >
      control;  //!< List of control parametrization data
  std::vector<Scalar> integral;
  VectorXs dx;               //!< State rate
  std::vector<VectorXs> ki;  //!< List of RK4 terms related to system dynamics
  std::vector<VectorXs>
      y;  //!< List of states where f is evaluated in the RK4 integration
  std::vector<VectorXs>
      ws;  //!< Control inputs evaluated in the RK4 integration
  std::vector<VectorXs> dx_rk4;

  std::vector<MatrixXs>
      dki_dx;  //!< List of partial derivatives of RK4 nodes with respect to the
               //!< state of the RK4 integration. dki/dx
  std::vector<MatrixXs>
      dki_du;  //!< List of partial derivatives of RK4 nodes with respect to the
               //!< control parameters of the RK4 integration. dki/du

  std::vector<MatrixXs>
      dyi_dx;  //!< List of partial derivatives of RK4 dynamics with respect to
               //!< the state of the RK4 integrator. dyi/dx
  std::vector<MatrixXs>
      dyi_du;  //!< List of partial derivatives of RK4 dynamics with respect to
               //!< the control parameters of the RK4 integrator. dyi/du

  std::vector<VectorXs>
      dli_dx;  //!< List of partial derivatives of the cost with respect to the
               //!< state of the RK4 integration. dli_dx
  std::vector<VectorXs>
      dli_du;  //!< List of partial derivatives of the cost with respect to the
               //!< control input of the RK4 integration. dli_du

  std::vector<MatrixXs>
      ddli_ddx;  //!< List of second partial derivatives of the cost with
                 //!< respect to the state of the RK4 integration. ddli_ddx
  std::vector<MatrixXs>
      ddli_ddw;  //!< List of second partial derivatives of the cost with
                 //!< respect to the control parameters of the RK4 integration.
                 //!< ddli_ddw
  std::vector<MatrixXs> ddli_ddu;  //!< List of second partial derivatives of
                                   //!< the cost with respect to the control
                                   //!< input of the RK4 integration. ddli_ddu
  std::vector<MatrixXs>
      ddli_dxdw;  //!< List of second partial derivatives of the cost with
                  //!< respect to the state and control input of the RK4
                  //!< integration. ddli_dxdw
  std::vector<MatrixXs>
      ddli_dxdu;  //!< List of second partial derivatives of the cost with
                  //!< respect to the state and control parameters of the RK4
                  //!< integration. ddli_dxdu
  std::vector<MatrixXs>
      ddli_dwdu;  //!< List of second partial derivatives of the cost with
                  //!< respect to the control parameters and inputs control of
                  //!< the RK4 integration. ddli_dwdu

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

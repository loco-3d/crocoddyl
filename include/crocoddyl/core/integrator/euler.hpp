///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {

/**
 * @brief Symplectic Euler integrator
 *
 * It applies a symplectic Euler integration scheme to a differential (i.e., continuous time) action model.
 *
 * This symplectic Euler scheme introduces also the possibility to parametrize the control trajectory inside an
 * integration step, for instance using polynomials. This requires introducing some notation to clarify the difference
 * between the control inputs of the differential model and the control inputs to the integrated model. We have decided
 * to use \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and \f$\mathbf{u}\f$ for the
 * control inputs of the integrated action model. Note that the zero-order (e.g.,
 * `ControlParametrizationModelPolyZeroTpl`) are the only ones that make sense to use within this integrator.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelEulerTpl : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> ControlParametrizationModelAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;
  
  /**
   * @brief Initialize the symplectic Euler integrator
   *
   * @param[in] model               Differential action model
   * @param[in] control             Control parametrization
   * @param[in] time_step           Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelEulerTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                boost::shared_ptr<ControlParametrizationModelAbstract> control,
                                const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);

  /**
   * @brief Initialize the symplectic Euler integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the control parametrization.
   *
   * @param[in] model               Differential action model
   * @param[in] time_step           Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelEulerTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  virtual ~IntegratedActionModelEulerTpl();

  /**
   * @brief Integrate the differential action model using symplectic Euler scheme
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Integrate the total cost value for nodes that depends only on the state using symplectic Euler scheme
   *
   * It computes the total cost and defines the next state as the current one. This function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the partial derivatives of the symplectic Euler integrator
   *
   * @param[in] data  Symplectic Euler data
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
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the symplectic Euler data
   *
   * @return the symplectic Euler data
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
   * @param[in] data    Symplectic Euler data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Compute the Sparse product between the given matrix A and the Jacobian of the dynamics with respect to the
   * control
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in]  Fu   Jacobian matrix of the dynamics w.r.t. control
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between A and the Jacobian of the dynamics with respect to the control
   * @param[in] op      Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp = setto) const;

  /**
   * @brief Compute the Sparse product between the Jacobian of the dynamics with respect to the
   * control and the given matrix A 
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in]  FuTranspose   Transpose of Jacobian matrix of the dynamics w.r.t. control
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between A and the Jacobian of the dynamics with respect to the control
   * @param[in] op      Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& FuTranspose, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXdRowMajor> out, const AssignmentOp = setto) const;
     
  /**
   * @brief Print relevant information of the Euler integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::control_;             //!< Control parametrization
  using Base::differential_;        //!< Differential action model
  using Base::nu_;                  //!< Dimension of the control
  using Base::state_;               //!< Model of the state
  using Base::time_step2_;          //!< Square of the time step used for integration
  using Base::time_step_;           //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual is used
};

template <typename _Scalar>
struct IntegratedActionDataEulerTpl : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataEulerTpl(Model<Scalar>* const model) : Base(model) {
    differential = model->get_differential()->createData();
    control = model->get_control()->createData();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    dx = VectorXs::Zero(ndx);
    da_du = MatrixXs::Zero(nv, model->get_nu());
    Lwu = MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu());
  }
  virtual ~IntegratedActionDataEulerTpl() {}

  boost::shared_ptr<DifferentialActionDataAbstract> differential;  //!< Differential model data
  boost::shared_ptr<ControlParametrizationDataAbstract> control;   //!< Control parametrization data
  VectorXs dx;
  MatrixXs da_du;
  MatrixXs Lwu;  //!< Hessian of the cost function with respect to the control input (w) and control parameters (u)

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

#endif  // CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

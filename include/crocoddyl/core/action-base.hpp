///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTION_BASE_HPP_
#define CROCODDYL_CORE_ACTION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for action model
 *
 * In Crocoddyl, an action model combines dynamics and cost models. Each node, in our optimal control problem, is
 * described through an action model. Every time that we want describe a problem, we need to provide ways of computing
 * the dynamics, cost functions and their derivatives. All these is described inside the action model.
 *
 * Concretely speaking, the action model describes a time-discrete action model with a first-order ODE along a cost
 * function, i.e.
 * \f[
 * \begin{aligned}
 * &\delta\mathbf{x}^+ = \mathbf{f_{x}}\delta\mathbf{x}+\mathbf{f_{u}}\delta\mathbf{u}, &\textrm{(dynamics)}\\
 * &l(\delta\mathbf{x},\delta\mathbf{u}) = \begin{bmatrix}1 \\ \delta\mathbf{x} \\ \delta\mathbf{u}\end{bmatrix}^T
 * \begin{bmatrix}0 & \mathbf{l_x}^T & \mathbf{l_u}^T \\ \mathbf{l_x} & \mathbf{l_{xx}} & \mathbf{l_{ux}}^T \\
 * \mathbf{l_u} & \mathbf{l_{ux}} & \mathbf{l_{uu}}\end{bmatrix} \begin{bmatrix}1 \\ \delta\mathbf{x} \\
 * \delta\mathbf{u}\end{bmatrix}, &\textrm{(cost)}
 * \end{aligned}
 * \f]
 * where the state \f$\mathbf{x}\in\mathcal{X}\f$ lies in the state manifold
 * described with a `nx`-tuple, its rate \f$\delta\mathbf{x}\in T_{\mathbf{x}}\mathcal{X}\f$ is a tangent vector to
 * this manifold with `ndx` dimension, and \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ is the input commands. Note that the we
 * could describe a linear or linearized action system, where the cost has a quadratic form.
 *
 * The main computations are carrying out in `calc` and `calcDiff`. `calc` computes the next state and cost and
 * `calcDiff` computes the derivatives of the dynamics and cost function. Concretely speaking, `calcDiff` builds a
 * linear-quadratic approximation of an action model, where the dynamics and cost functions have linear and
 * quadratic forms, respectively. \f$\mathbf{f_x}\in\mathbb{R}^{nv\times ndx}\f$,
 * \f$\mathbf{f_u}\in\mathbb{R}^{nv\times nu}\f$ are the Jacobians of the dynamics;
 * \f$\mathbf{l_x}\in\mathbb{R}^{ndx}\f$, \f$\mathbf{l_u}\in\mathbb{R}^{nu}\f$,
 * \f$\mathbf{l_{xx}}\in\mathbb{R}^{ndx\times ndx}\f$, \f$\mathbf{l_{xu}}\in\mathbb{R}^{ndx\times nu}\f$,
 * \f$\mathbf{l_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the Jacobians and Hessians of the cost function, respectively.
 * Additionally, it is important remark that `calcDiff()` computes the derivates using the latest stored values by
 * `calc()`. Thus, we need to run first `calc()`.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActionModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the action model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of control vector
   * @param[in] nr     Dimension of cost-residual vector
   */
  ActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t nu, const std::size_t nr = 0);
  virtual ~ActionModelAbstractTpl();

  /**
   * @brief Compute the next state and cost value
   *
   * @param[in] data  Action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost function. It assumes that `calc()` has
   * been run first. This function builds a linear-quadratic approximation of the action model (i.e. dynamical system
   * and cost function).
   *
   * @param[in] data  Action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Create the action data
   *
   * @return the action data
   */
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  /**
   * @copybrief calc()
   *
   * @param[in] data  Action data
   * @param[in] x     State point
   */
  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @copybrief calcDiff()
   *
   * @param[in] data  Action data
   * @param[in] x     State point
   */
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference posture as an equilibrium point, i.e.
   * for \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data    Action data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @copybrief quasicStatic()
   *
   * @copydetails quasicStatic()
   *
   * @param[in] data    Action data
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   * @return Quasic static commands
   */
  VectorXs quasiStatic_x(const boost::shared_ptr<ActionDataAbstract>& data, const VectorXs& x,
                         const std::size_t maxiter = 100, const Scalar tol = Scalar(1e-9));

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the dimension of the cost-residual vector
   */
  std::size_t get_nr() const;

  /**
   * @brief Return the state
   */
  const boost::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the control lower bound
   */
  const VectorXs& get_u_lb() const;

  /**
   * @brief Return the control upper bound
   */
  const VectorXs& get_u_ub() const;

  /**
   * @brief Indicates if there are defined control limits
   */
  bool get_has_control_limits() const;

  /**
   * @brief Modify the control lower bounds
   */
  void set_u_lb(const VectorXs& u_lb);

  /**
   * @brief Modify the control upper bounds
   */
  void set_u_ub(const VectorXs& u_ub);

  /**
   * @brief Print information on the ActionModel
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const ActionModelAbstractTpl<Scalar>& action_model);

 protected:
  /**
   * @brief Print relevant information of the action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

  std::size_t nu_;                          //!< Control dimension
  std::size_t nr_;                          //!< Dimension of the cost residual
  boost::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;                          //!< Neutral state
  VectorXs u_lb_;                           //!< Lower control limits
  VectorXs u_ub_;                           //!< Upper control limits
  bool has_control_limits_;                 //!< Indicates whether any of the control limits is finite

  /**
   * @brief Update the status of the control limits (i.e. if there are defined limits)
   */
  void update_has_control_limits();
};

template <typename _Scalar>
struct ActionDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActionDataAbstractTpl(Model<Scalar>* const model)
      : cost(Scalar(0.)),
        xnext(model->get_state()->get_nx()),
        Fx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Fu(model->get_state()->get_ndx(), model->get_nu()),
        r(model->get_nr()),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()) {
    xnext.setZero();
    Fx.setZero();
    Fu.setZero();
    r.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }
  virtual ~ActionDataAbstractTpl() {}

  Scalar cost;     //!< cost value
  VectorXs xnext;  //!< evolution state
  MatrixXs Fx;     //!< Jacobian of the dynamics
  MatrixXs Fu;     //!< Jacobian of the dynamics
  VectorXs r;      //!< Cost residual
  VectorXs Lx;     //!< Jacobian of the cost function
  VectorXs Lu;     //!< Jacobian of the cost function
  MatrixXs Lxx;    //!< Hessian of the cost function
  MatrixXs Lxu;    //!< Hessian of the cost function
  MatrixXs Luu;    //!< Hessian of the cost function
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/action-base.hxx"

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_

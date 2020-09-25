///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
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
 * &\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k,\mathbf{u}_k), &\textrm{(dynamics)}\\
 * &l(\mathbf{x}_k,\mathbf{u}_k) = a(\mathbf{r}(\mathbf{x}_k,\mathbf{u}_k)), &\textrm{(cost)}\\
 * &\mathbf{g}(\mathbf{x}_k,\mathbf{u}_k)<\mathbf{0}, &\textrm{(inequality constraints)}\\
 * &\mathbf{h}(\mathbf{x}_k,\mathbf{u}_k)=\mathbf{0}, &\textrm{(equality constraints)}
 * \end{aligned}
 * \f]
 * where
 *  - the state \f$\mathbf{x}\in\mathcal{X}\f$ lies in the state manifold described with a `nx`-tuple and its tangent
 * vector is defined as \f$\delta\mathbf{x}\in T_{\mathbf{x}}\mathcal{X}\f$ (vector with `ndx` dimension),
 *  - the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ is an Euclidean vector
 *  - \f$\mathbf{r}(\cdot)\f$ and \f$a(\cdot)\f$ are the residual and activation functions (see
 * `ActivationModelAbstractTpl`),
 *  - \f$\mathbf{g}(\cdot)\in\mathbb{R}^{ng}\f$ and \f$\mathbf{h}(\cdot)\in\mathbb{R}^{nh}\f$ are the inequality and
 * equality vector functions, respectively.
 *
 * The computation of these equations are carrying out inside `calc()` function. In short, this function computes the
 * system acceleration, cost and constraints values (also called constraints violations). This procedure is equivalent
 * to running a forward pass of the action model.
 *
 * However, during numerical optimization, we also need to run backward passes of the action model. These calculations
 * are performed by `calcDiff()`. In short, this function builds a linear-quadratic approximation of the action model,
 * i.e.: \f[ \begin{aligned}
 * &\delta\mathbf{x}_{k+1} = \mathbf{f_x}\delta\mathbf{x}_k+\mathbf{f_u}\delta\mathbf{u}_k, &\textrm{(dynamics)}\\
 * &l(\delta\mathbf{x}_k,\delta\mathbf{u}_k) = \begin{bmatrix}1 \\ \delta\mathbf{x}_k \\
 * \delta\mathbf{u}_k\end{bmatrix}^T
 * \begin{bmatrix}0 & \mathbf{l_x}^T & \mathbf{l_u}^T \\ \mathbf{l_x} & \mathbf{l_{xx}} & \mathbf{l_{ux}}^T \\
 * \mathbf{l_u} & \mathbf{l_{ux}} & \mathbf{l_{uu}}\end{bmatrix} \begin{bmatrix}1 \\ \delta\mathbf{x}_k \\
 * \delta\mathbf{u}_k\end{bmatrix}, &\textrm{(cost)}\\
 * &\mathbf{g}(\delta\mathbf{x}_k,\delta\mathbf{u}_k)<\mathbf{0}, &\textrm{(inequality constraint)}\\
 * &\mathbf{h}(\delta\mathbf{x}_k,\delta\mathbf{u}_k)=\mathbf{0}, &\textrm{(equality constraint)}
 * \end{aligned}
 * \f]
 * where
 *  - \f$\mathbf{f_x}\in\mathbb{R}^{ndx\times ndx}\f$ and \f$\mathbf{f_u}\in\mathbb{R}^{ndx\times nu}\f$ are the
 * Jacobians of the dynamics,
 *  - \f$\mathbf{l_x}\in\mathbb{R}^{ndx}\f$ and \f$\mathbf{l_u}\in\mathbb{R}^{nu}\f$ are the Jacobians of the cost
 * function,
 *  - \f$\mathbf{l_{xx}}\in\mathbb{R}^{ndx\times ndx}\f$, \f$\mathbf{l_{xu}}\in\mathbb{R}^{ndx\times nu}\f$ and
 * \f$\mathbf{l_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the Hessians of the cost function,
 *  - \f$\mathbf{g_x}\in\mathbb{R}^{ng\times ndx}\f$ and \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$ are the
 * Jacobians of the inequality constraints, and
 *  - \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$ and \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$ are the
 * Jacobians of the equality constraints.
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
   * @param[in] ng     Number of inequality constraints
   * @param[in] nh     Number of equality constraints
   */
  ActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t& nu, const std::size_t& nr = 0,
                         const std::size_t& ng = 0, const std::size_t& nh = 0);
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
                           const Eigen::Ref<const VectorXs>& x, const std::size_t& maxiter = 100,
                           const Scalar& tol = Scalar(1e-9));

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
                         const std::size_t& maxiter = 100, const Scalar& tol = Scalar(1e-9));

  /**
   * @brief Return the dimension of the control input
   */
  const std::size_t& get_nu() const;

  /**
   * @brief Return the dimension of the cost-residual vector
   */
  const std::size_t& get_nr() const;

  /**
   * @brief Return the number of inequality constraints
   */
  const std::size_t& get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  const std::size_t& get_nh() const;

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
  bool const& get_has_control_limits() const;

  /**
   * @brief Modify the control lower bounds
   */
  void set_u_lb(const VectorXs& u_lb);

  /**
   * @brief Modify the control upper bounds
   */
  void set_u_ub(const VectorXs& u_ub);

 protected:
  std::size_t nu_;                          //!< Control dimension
  std::size_t nr_;                          //!< Dimension of the cost residual
  std::size_t ng_;                          //!< Number of inequality constraints
  std::size_t nh_;                          //!< Number of equality constraints
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
        Luu(model->get_nu(), model->get_nu()),
        g(model->get_ng()),
        Gx(model->get_ng(), model->get_state()->get_ndx()),
        Gu(model->get_ng(), model->get_nu()),
        h(model->get_nh()),
        Hx(model->get_nh(), model->get_state()->get_ndx()),
        Hu(model->get_nh(), model->get_nu()) {
    xnext.setZero();
    Fx.setZero();
    Fu.setZero();
    r.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
    g.setZero();
    Gx.setZero();
    Gu.setZero();
    h.setZero();
    Hx.setZero();
    Hu.setZero();
  }
  virtual ~ActionDataAbstractTpl() {}

  Scalar cost;     //!< cost value
  VectorXs xnext;  //!< evolution state
  MatrixXs Fx;     //!< Jacobian of the dynamics
  MatrixXs Fu;     //!< Jacobian of the dynamics
  VectorXs r;      //!< Cost residual
  VectorXs Lx;     //!< Jacobian of the cost
  VectorXs Lu;     //!< Jacobian of the cost
  MatrixXs Lxx;    //!< Hessian of the cost
  MatrixXs Lxu;    //!< Hessian of the cost
  MatrixXs Luu;    //!< Hessian of the cost
  VectorXs g;      //!< Inequality constraint values
  MatrixXs Gx;     //!< Jacobian of the inequality constraint
  MatrixXs Gu;     //!< Jacobian of the inequality constraint
  VectorXs h;      //!< Equality constraint values
  MatrixXs Hx;     //!< Jacobian of the equality constraint
  MatrixXs Hu;     //!< Jacobian of the equality constraint
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/action-base.hxx"

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_

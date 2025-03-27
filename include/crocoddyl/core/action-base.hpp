///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTION_BASE_HPP_
#define CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

class ActionModelBase {
 public:
  virtual ~ActionModelBase() = default;

  CROCODDYL_BASE_CAST(ActionModelBase, ActionModelAbstractTpl)
};

/**
 * @brief Abstract class for action model
 *
 * An action model combines dynamics, cost functions and constraints. Each node,
 * in our optimal control problem, is described through an action model. Every
 * time that we want describe a problem, we need to provide ways of computing
 * the dynamics, cost functions, constraints and their derivatives. All these is
 * described inside the action model.
 *
 * Concretely speaking, the action model describes a time-discrete action model
 * with a first-order ODE along a cost function, i.e.
 *  - the state \f$\mathbf{z}\in\mathcal{Z}\f$ lies in a manifold described with
 * a `nx`-tuple,
 *  - the state rate \f$\mathbf{\dot{x}}\in T_{\mathbf{q}}\mathcal{Q}\f$ is the
 * tangent vector to the state manifold with `ndx` dimension,
 *  - the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ is an Euclidean
 * vector
 *  - \f$\mathbf{r}(\cdot)\f$ and \f$a(\cdot)\f$ are the residual and activation
 * functions (see `ResidualModelAbstractTpl` and `ActivationModelAbstractTpl`,
 * respetively),
 *  - \f$\mathbf{g}(\cdot)\in\mathbb{R}^{ng}\f$ and
 * \f$\mathbf{h}(\cdot)\in\mathbb{R}^{nh}\f$ are the inequality and equality
 * vector functions, respectively.
 *
 * The computation of these equations are carried out out inside `calc()`
 * function. In short, this function computes the system acceleration, cost and
 * constraints values (also called constraints violations). This procedure is
 * equivalent to running a forward pass of the action model.
 *
 * However, during numerical optimization, we also need to run backward passes
 * of the action model. These calculations are performed by `calcDiff()`. In
 * short, this method builds a linear-quadratic approximation of the action
 * model, i.e.: \f[ \begin{aligned}
 * &\delta\mathbf{x}_{k+1} =
 * \mathbf{f_x}\delta\mathbf{x}_k+\mathbf{f_u}\delta\mathbf{u}_k,
 * &\textrm{(dynamics)}\\
 * &\ell(\delta\mathbf{x}_k,\delta\mathbf{u}_k) = \begin{bmatrix}1
 * \\ \delta\mathbf{x}_k \\ \delta\mathbf{u}_k\end{bmatrix}^T \begin{bmatrix}0 &
 * \mathbf{\ell_x}^T & \mathbf{\ell_u}^T \\ \mathbf{\ell_x} & \mathbf{\ell_{xx}}
 * &
 * \mathbf{\ell_{ux}}^T \\
 * \mathbf{\ell_u} & \mathbf{\ell_{ux}} & \mathbf{\ell_{uu}}\end{bmatrix}
 * \begin{bmatrix}1 \\ \delta\mathbf{x}_k \\
 * \delta\mathbf{u}_k\end{bmatrix}, &\textrm{(cost)}\\
 * &\mathbf{g}(\delta\mathbf{x}_k,\delta\mathbf{u}_k)<\mathbf{0},
 * &\textrm{(inequality constraint)}\\
 * &\mathbf{h}(\delta\mathbf{x}_k,\delta\mathbf{u}_k)=\mathbf{0},
 * &\textrm{(equality constraint)} \end{aligned} \f] where
 *  - \f$\mathbf{f_x}\in\mathbb{R}^{ndx\times ndx}\f$ and
 * \f$\mathbf{f_u}\in\mathbb{R}^{ndx\times nu}\f$ are the Jacobians of the
 * dynamics,
 *  - \f$\mathbf{\ell_x}\in\mathbb{R}^{ndx}\f$ and
 * \f$\mathbf{\ell_u}\in\mathbb{R}^{nu}\f$ are the Jacobians of the cost
 * function,
 *  - \f$\mathbf{\ell_{xx}}\in\mathbb{R}^{ndx\times ndx}\f$,
 * \f$\mathbf{\ell_{xu}}\in\mathbb{R}^{ndx\times nu}\f$ and
 * \f$\mathbf{\ell_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the Hessians of the
 * cost function,
 *  - \f$\mathbf{g_x}\in\mathbb{R}^{ng\times ndx}\f$ and
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$ are the Jacobians of the
 * inequality constraints, and
 *  - \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$ and
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$ are the Jacobians of the
 * equality constraints.
 *
 * Additionally, it is important to note that `calcDiff()` computes the
 * derivatives using the latest stored values by `calc()`. Thus, we need to
 * first run `calc()`.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActionModelAbstractTpl : public ActionModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
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
   * @param[in] ng     Number of inequality constraints (default 0)
   * @param[in] nh     Number of equality constraints (default 0)
   * @param[in] ng_T   Number of inequality terminal constraints (default 0)
   * @param[in] nh_T   Number of equality terminal constraints (default 0)
   */
  ActionModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                         const std::size_t nu, const std::size_t nr = 0,
                         const std::size_t ng = 0, const std::size_t nh = 0,
                         const std::size_t ng_T = 0,
                         const std::size_t nh_T = 0);
  /**
   * @brief Copy constructor
   * @param other  Action model to be copied
   */
  ActionModelAbstractTpl(const ActionModelAbstractTpl<Scalar>& other);

  virtual ~ActionModelAbstractTpl() = default;

  /**
   * @brief Compute the next state and cost value
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the total cost value for nodes that depends only on the
   * state
   *
   * It updates the total cost and the next state is not computed as it is not
   * expected to change. This function is used in the terminal nodes of an
   * optimal control problem.
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost
   * function. It assumes that `calc()` has been run first. This function builds
   * a linear-quadratic approximation of the action model (i.e. dynamical system
   * and cost function).
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the derivatives of the cost functions with respect to the
   * state only
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the action data
   *
   * @return the action data
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
   * @param[in] data    Action data
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
  VectorXs quasiStatic_x(const std::shared_ptr<ActionDataAbstract>& data,
                         const VectorXs& x, const std::size_t maxiter = 100,
                         const Scalar tol = Scalar(1e-9));

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the dimension of the cost-residual vector
   */
  std::size_t get_nr() const;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const;

  /**
   * @brief Return the number of inequality terminal constraints
   */
  virtual std::size_t get_ng_T() const;

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_nh_T() const;

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const;

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
   * @brief Modify the lower bound of the inequality constraints
   */
  void set_g_lb(const VectorXs& g_lb);

  /**
   * @brief Modify the upper bound of the inequality constraints
   */
  void set_g_ub(const VectorXs& g_ub);

  /**
   * @brief Modify the control lower bounds
   */
  void set_u_lb(const VectorXs& u_lb);

  /**
   * @brief Modify the control upper bounds
   */
  void set_u_ub(const VectorXs& u_ub);

  /**
   * @brief Print information on the action model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ActionModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::size_t nu_;    //!< Control dimension
  std::size_t nr_;    //!< Dimension of the cost residual
  std::size_t ng_;    //!< Number of inequality constraints
  std::size_t nh_;    //!< Number of equality constraints
  std::size_t ng_T_;  //!< Number of inequality terminal constraints
  std::size_t nh_T_;  //!< Number of equality terminal constraints
  std::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;                        //!< Neutral state
  VectorXs g_lb_;            //!< Lower bound of the inequality constraints
  VectorXs g_ub_;            //!< Lower bound of the inequality constraints
  VectorXs u_lb_;            //!< Lower control limits
  VectorXs u_ub_;            //!< Upper control limits
  bool has_control_limits_;  //!< Indicates whether any of the control limits is
                             //!< finite
  ActionModelAbstractTpl()
      : nu_(0), nr_(0), ng_(0), nh_(0), ng_T_(0), nh_T_(0), state_(nullptr) {}

  /**
   * @brief Update the status of the control limits (i.e. if there are defined
   * limits)
   */
  void update_has_control_limits();

  template <class Scalar>
  friend class ConstraintModelManagerTpl;
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
        g(model->get_ng() > model->get_ng_T() ? model->get_ng()
                                              : model->get_ng_T()),
        Gx(model->get_ng() > model->get_ng_T() ? model->get_ng()
                                               : model->get_ng_T(),
           model->get_state()->get_ndx()),
        Gu(model->get_ng() > model->get_ng_T() ? model->get_ng()
                                               : model->get_ng_T(),
           model->get_nu()),
        h(model->get_nh() > model->get_nh_T() ? model->get_nh()
                                              : model->get_nh_T()),
        Hx(model->get_nh() > model->get_nh_T() ? model->get_nh()
                                               : model->get_nh_T(),
           model->get_state()->get_ndx()),
        Hu(model->get_nh() > model->get_nh_T() ? model->get_nh()
                                               : model->get_nh_T(),
           model->get_nu()) {
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
  virtual ~ActionDataAbstractTpl() = default;

  Scalar cost;     //!< cost value
  VectorXs xnext;  //!< evolution state
  MatrixXs Fx;  //!< Jacobian of the dynamics w.r.t. the state \f$\mathbf{x}\f$
  MatrixXs
      Fu;      //!< Jacobian of the dynamics w.r.t. the control \f$\mathbf{u}\f$
  VectorXs r;  //!< Cost residual
  VectorXs Lx;   //!< Jacobian of the cost w.r.t. the state \f$\mathbf{x}\f$
  VectorXs Lu;   //!< Jacobian of the cost w.r.t. the control \f$\mathbf{u}\f$
  MatrixXs Lxx;  //!< Hessian of the cost w.r.t. the state \f$\mathbf{x}\f$
  MatrixXs Lxu;  //!< Hessian of the cost w.r.t. the state \f$\mathbf{x}\f$ and
                 //!< control \f$\mathbf{u}\f$
  MatrixXs Luu;  //!< Hessian of the cost w.r.t. the control \f$\mathbf{u}\f$
  VectorXs g;    //!< Inequality constraint values
  MatrixXs Gx;   //!< Jacobian of the inequality constraint w.r.t. the state
                 //!< \f$\mathbf{x}\f$
  MatrixXs Gu;   //!< Jacobian of the inequality constraint w.r.t. the control
                 //!< \f$\mathbf{u}\f$
  VectorXs h;    //!< Equality constraint values
  MatrixXs Hx;   //!< Jacobian of the equality constraint w.r.t. the state
                 //!< \f$\mathbf{x}\f$
  MatrixXs Hu;   //!< Jacobian of the equality constraint w.r.t. the control
                 //!< \f$\mathbf{u}\f$
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/action-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActionModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActionDataAbstractTpl)

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_

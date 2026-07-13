///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DYNAMICS_BASE_HPP_
#define CROCODDYL_CORE_DYNAMICS_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

/**
 * @brief Type of dynamics behavior
 *
 * Defines how the dynamics model behaves when used inside an integrated action
 * model:
 *  - `ContinuousControl`: continuous-time dynamics for control purposes; in
 *     inverse dynamics uses the condensed (reduced) formulation,
 *  - `ContinuousEstimation`: continuous-time dynamics for estimation purposes;
 *     in inverse dynamics uses the full dynamics formulation,
 *  - `DiscreteTime`: discrete-time dynamics (e.g., impulse dynamics).
 */
enum class DynamicsType {
  ContinuousControl,  //!< Continuous dynamics for control (condensed inverse)
  ContinuousEstimation,  //!< Continuous dynamics for estimation (full inverse)
  DiscreteTime           //!< Discrete-time dynamics (impulse dynamics)
};

class DynamicsModelBase {
 public:
  virtual ~DynamicsModelBase() = default;

  CROCODDYL_BASE_CAST(DynamicsModelBase, DynamicsModelAbstractTpl)
};

/**
 * @brief Abstract class for dynamics models
 *
 * A dynamics model defines the continuous-time ODE (or discrete-time map) of a
 * dynamical system, decoupled from any integration scheme. It is parametrized
 * by a state \f$\mathbf{x}\f$, a control \f$\mathbf{u}\f$, and optionally a
 * parameter vector \f$\mathbf{p}\in\mathbb{R}^{np}\f$.
 *
 * Concretely speaking, the dynamics model computes the system acceleration
 * \f$\dot{\mathbf{v}}\f$ and, if defined, the equality and inequality
 * constraint residuals:
 * \f[
 *   \begin{aligned}
 *     &\dot{\mathbf{v}} = \mathbf{f}(\mathbf{q}, \mathbf{v}, \mathbf{u},
 *       \mathbf{p}), &\textrm{(dynamics)}\\
 *     &\mathbf{g}(\mathbf{q}, \mathbf{v}, \mathbf{u}, \mathbf{p}) <
 *       \mathbf{0}, &\textrm{(inequality constraints)}\\
 *     &\mathbf{h}(\mathbf{q}, \mathbf{v}, \mathbf{u}, \mathbf{p}) =
 *       \mathbf{0}, &\textrm{(equality constraints)}
 *   \end{aligned}
 * \f]
 *
 * The running-node `calcDiff()` method dispatches to `calcDiff_xu()` and, when
 * \f$n_p > 0\f$, also to `calcDiff_p()`, building a first-order approximation:
 * \f[
 *   \delta\dot{\mathbf{v}} =
 *     \mathbf{F_x}\delta\mathbf{x} +
 *     \mathbf{F_u}\delta\mathbf{u} +
 *     \mathbf{F_p}\delta\mathbf{p}.
 * \f]
 *
 * Parameterized models first attach their manager through `set_params()`,
 * then update the active parameter vector through `update_p()` before calling
 * `calc()` or `calcDiff()`.
 *
 * \sa `calc()`, `calcDiff()`, `calcDiff_xu()`, `calcDiff_p()`, `createData()`
 */
template <typename _Scalar>
class DynamicsModelAbstractTpl : public DynamicsModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the dynamics model
   *
   * @param[in] state     State description
   * @param[in] dyn_type  Type of dynamics behavior (default ContinuousControl)
   * @param[in] np        Dimension of the parameter vector (default 0)
   * @param[in] nu        Dimension of control vector (default 0)
   * @param[in] ng        Number of inequality constraints (default 0)
   * @param[in] nh        Number of equality constraints (default 0)
   */
  DynamicsModelAbstractTpl(
      std::shared_ptr<StateAbstract> state,
      const DynamicsType dyn_type = DynamicsType::ContinuousControl,
      const std::size_t np = 0, const std::size_t nu = 0,
      const std::size_t ng = 0, const std::size_t nh = 0);
  virtual ~DynamicsModelAbstractTpl() = default;

  /**
   * @brief Compute the system acceleration
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the system acceleration for terminal nodes (no control)
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the dynamics
   *
   * Dispatches to `calcDiff_xu()` and, when \f$n_p > 0\f$, to
   * `calcDiff_p()`. It assumes that `calc()` has been run first.
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the state derivatives for terminal nodes (no control)
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the dynamics w.r.t. state and control
   *
   * Stores results in `data->Fx` and `data->Fu` and the constraint Jacobians
   * `data->Hx`, `data->Hu`, `data->Gx`, `data->Gu`.
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the derivatives of the dynamics w.r.t. state only (terminal)
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the parameter sensitivity of the dynamics
   *
   * Stores results in `data->Fp`, `data->Hp`, `data->Gp`. This method must be
   * overridden by derived classes when \f$n_p > 0\f$.
   *
   * @param[in] data  Dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the dynamics data
   *
   * @return the dynamics data
   */
  virtual std::shared_ptr<DynamicsDataAbstract> createData();

  /**
   * @brief Create dynamics data with optional external parameter data
   *
   * The default implementation forwards to zero-argument `createData()`.
   * Parameterized dynamics models override this overload to attach shared
   * parameter data to their concrete dynamics data.
   *
   * @param[in] params_data  Optional external parameter-manager data
   * @return the dynamics data
   */
  virtual std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data);

  /**
   * @brief Wire a parameter manager into this dynamics model and its data
   *
   * Parameterized dynamics models override this hook to attach their manager.
   * The abstract default fails fast because generic dynamics data has no
   * parameter-manager payload.
   *
   * @param[in] data    Dynamics data
   * @param[in] params  Parameter manager
   */
  virtual void set_params(const std::shared_ptr<DynamicsDataAbstract>& data,
                          std::shared_ptr<ParameterManager> params);

  /**
   * @brief Update the active parameter vector inside the given dynamics data
   *
   * Parameterized dynamics models override this hook when they support
   * parameter updates. The abstract default fails fast.
   *
   * @param[in] data  Dynamics data
   * @param[in] p     Parameter vector \f$\mathbf{p}\in\mathbb{R}^{np}\f$
   */
  virtual void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p);

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const std::shared_ptr<DynamicsDataAbstract>& data);

  /**
   * @brief Computes the quasi-static commands
   *
   * The quasi-static commands are the ones produced for the reference posture
   * as an equilibrium point, i.e.
   * \f$\mathbf{f}(\mathbf{q},\mathbf{v}=\mathbf{0},\mathbf{u})=\mathbf{0}\f$.
   *
   * @param[in] data    Dynamics data
   * @param[out] u      Quasi-static commands
   * @param[in] x       State point (velocity must be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<DynamicsDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @copybrief quasiStatic()
   *
   * @copydetails quasiStatic()
   *
   * @param[in] data    Dynamics data
   * @param[in] x       State point (velocity must be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   * @return Quasi-static commands
   */
  VectorXs quasiStatic_x(const std::shared_ptr<DynamicsDataAbstract>& data,
                         const VectorXs& x, const std::size_t maxiter = 100,
                         const Scalar tol = Scalar(1e-9));

  /**
   * @brief Update the torque measurements
   *
   * @param[in] tau_meas  Measured torques (dim. nu)
   */
  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas);

  /**
   * @brief Set the dynamics type
   *
   * @param[in] dyn_type  Type of dynamics behavior
   */
  void set_dyn_type(const DynamicsType dyn_type);

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the dimension of the parameter vector
   */
  std::size_t get_np() const;

  /**
   * @brief Return the number of inequality constraints
   */
  std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  std::size_t get_nh() const;

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the dynamics type
   */
  DynamicsType get_dyn_type() const;

  /**
   * @brief Return the measured torques
   */
  const VectorXs& get_tau_meas() const;

  /**
   * @brief Return the parameter lower bounds
   */
  const VectorXs& get_p_lb() const;

  /**
   * @brief Return the parameter upper bounds
   */
  const VectorXs& get_p_ub() const;

  /**
   * @brief Modify the parameter lower bounds
   */
  void set_p_lb(const VectorXs& p_lb);

  /**
   * @brief Modify the parameter upper bounds
   */
  void set_p_ub(const VectorXs& p_ub);

  /**
   * @brief Print information on the dynamics model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const DynamicsModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::size_t nu_;                        //!< Control dimension
  std::size_t np_;                        //!< Parameter dimension
  std::size_t ng_;                        //!< Number of inequality constraints
  std::size_t nh_;                        //!< Number of equality constraints
  std::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;         //!< Zero control vector (used for terminal nodes)
  VectorXs tau_meas_;      //!< Measured torques
  VectorXs p_lb_;          //!< Parameter lower bounds
  VectorXs p_ub_;          //!< Parameter upper bounds
  DynamicsType dyn_type_;  //!< Type of dynamics behavior

  DynamicsModelAbstractTpl()
      : nu_(0),
        np_(0),
        ng_(0),
        nh_(0),
        state_(nullptr),
        dyn_type_(DynamicsType::ContinuousControl) {}
};

/**
 * @brief Data structure for dynamics models
 *
 * The data stores the dynamics value and its state, control and parameter
 * Jacobians, equality and inequality constraint values/Jacobians, measured
 * dissipative power and its derivatives, and quasi-static workspace. For
 * continuous dynamics, `vdot` and the rows of `Fx`, `Fu` and `Fp` have
 * dimension `nv`; for discrete dynamics, `vdot` has dimension `nx` and the
 * Jacobian rows have dimension `ndx`. The optional `shared` collector is
 * non-owning and must outlive the data while attached.
 */
template <typename _Scalar>
struct DynamicsDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the dynamics data
   *
   * @param[in] model  Dynamics model
   */
  template <template <typename Scalar> class Model>
  explicit DynamicsDataAbstractTpl(Model<Scalar>* const model) {
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nq = model->get_state()->get_nq();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const bool is_discrete =
        (model->get_dyn_type() == DynamicsType::DiscreteTime);
    // For DiscreteTime: vdot stores the full next state (nq + nv = nx);
    //                   Jacobian rows use the tangent space dimension (ndx).
    // For continuous:   vdot stores the acceleration (nv);
    //                   Jacobian rows also use nv.
    const std::size_t nvdot = is_discrete ? nq + nv : nv;
    const std::size_t ndvdot = is_discrete ? ndx : nv;

    vdot.resize(nvdot);
    Fx.resize(ndvdot, ndx);
    Fu.resize(ndvdot, nu);
    Fp.resize(ndvdot, np);
    dissipative_P.resize(1);
    dP_dv.resize(1, nv);
    dP_dp.resize(1, np);
    h.resize(nh);
    Hx.resize(nh, ndx);
    Hu.resize(nh, nu);
    Hp.resize(nh, np);
    g.resize(ng);
    Gx.resize(ng, ndx);
    Gu.resize(ng, nu);
    Gp.resize(ng, np);
    tmp_ustatic.resize(nu);

    shared = NULL;
    setZero();
  }
  virtual ~DynamicsDataAbstractTpl() = default;

  /**
   * @brief Set all dynamics, derivative, constraint and workspace values to
   * zero
   *
   * Dimensions and the non-owning shared collector are left unchanged.
   */
  void setZero() {
    vdot.setZero();
    Fx.setZero();
    Fu.setZero();
    Fp.setZero();
    dissipative_P.setZero();
    dP_dv.setZero();
    dP_dp.setZero();
    h.setZero();
    Hx.setZero();
    Hu.setZero();
    Hp.setZero();
    g.setZero();
    Gx.setZero();
    Gu.setZero();
    Gp.setZero();
    tmp_ustatic.setZero();
  }

  DataCollectorAbstract* shared;  //!< Non-owning shared cost/constraint data
  VectorXs vdot;  //!< System acceleration (or next state for DiscreteTime)
  MatrixXs Fx;    //!< Jacobian of the dynamics w.r.t. the state
                  //!< \f$\mathbf{x}\f$
  MatrixXs Fu;    //!< Jacobian of the dynamics w.r.t. the control
                  //!< \f$\mathbf{u}\f$
  MatrixXs Fp;    //!< Jacobian of the dynamics w.r.t. the parameters
                  //!< \f$\mathbf{p}\f$
  VectorXs dissipative_P;  //!< Dissipative power
  MatrixXs dP_dv;          //!< Dissipative-power Jacobian w.r.t. velocity
  MatrixXs dP_dp;          //!< Dissipative-power Jacobian w.r.t. parameters
  VectorXs h;              //!< Equality constraint values
  MatrixXs Hx;  //!< Jacobian of equality constraints w.r.t. the state
  MatrixXs Hu;  //!< Jacobian of equality constraints w.r.t. the control
  MatrixXs Hp;  //!< Jacobian of equality constraints w.r.t. the parameters
  VectorXs g;   //!< Inequality constraint values
  MatrixXs Gx;  //!< Jacobian of inequality constraints w.r.t. the state
  MatrixXs Gu;  //!< Jacobian of inequality constraints w.r.t. the control
  MatrixXs Gp;  //!< Jacobian of inequality constraints w.r.t. the parameters
  VectorXs tmp_ustatic;  //!< Temporary vector for quasi-static computation
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/dynamics-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::DynamicsModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DynamicsDataAbstractTpl)

#endif  // CROCODDYL_CORE_DYNAMICS_BASE_HPP_

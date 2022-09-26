///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/math.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for differential action model
 *
 * A differential action model combines dynamics, cost and constraints models. We can use it in each node
 * of our optimal control problem thanks to dedicated integration rules (e.g., `IntegratedActionModelEulerTpl` or
 * `IntegratedActionModelRK4Tpl`). These integrated action models produces action models (`ActionModelAbstractTpl`).
 * Thus, every time that we want describe a problem, we need to provide ways of computing the dynamics, cost,
 * constraints functions and their derivatives. All these is described inside the differential action model.
 *
 * Concretely speaking, the differential action model is the time-continuous version of an action model, i.e.,
 * \f[
 * \begin{aligned}
 * &\dot{\mathbf{v}} = \mathbf{f}(\mathbf{q}, \mathbf{v}, \mathbf{u}), &\textrm{(dynamics)}\\
 * &\ell(\mathbf{q}, \mathbf{v},\mathbf{u}) = \int_0^{\delta t} a(\mathbf{r}(\mathbf{q}, \mathbf{v},\mathbf{u}))\,dt,
 * &\textrm{(cost)}\\
 * &\mathbf{g}(\mathbf{q}, \mathbf{v},\mathbf{u})<\mathbf{0}, &\textrm{(inequality constraint)}\\
 * &\mathbf{h}(\mathbf{q}, \mathbf{v},\mathbf{u})=\mathbf{0}, &\textrm{(equality constraint)}
 * \end{aligned}
 * \f]
 * where
 *  - the configuration \f$\mathbf{q}\in\mathcal{Q}\f$ lies in the configuration manifold described with a `nq`-tuple,
 *  - the velocity \f$\mathbf{v}\in T_{\mathbf{q}}\mathcal{Q}\f$ is the tangent vector to the configuration manifold
 * with `nv` dimension,
 *  - the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ is an Euclidean vector,
 *  - \f$\mathbf{r}(\cdot)\f$ and \f$a(\cdot)\f$ are the residual and activation functions (see
 * `ResidualModelAbstractTpl` and `ActivationModelAbstractTpl`, respectively),
 *  - \f$\mathbf{g}(\cdot)\in\mathbb{R}^{ng}\f$ and \f$\mathbf{h}(\cdot)\in\mathbb{R}^{nh}\f$ are the inequality and
 * equality vector functions, respectively.
 *
 * Both configuration and velocity describe the system space \f$\mathbf{x}=(\mathbf{q}, \mathbf{v})\in\mathcal{X}\f$
 * which lies in the state manifold. Note that the acceleration \f$\dot{\mathbf{v}}\in T_{\mathbf{q}}\mathcal{Q}\f$
 * lies also in the tangent space of the configuration manifold.
 * The computation of these equations are carrying out inside `calc()` function. In short, this function computes
 * the system acceleration, cost and constraints values (also called constraints violations). This procedure is
 * equivalent to running a forward pass of the action model.
 *
 * However, during numerical optimization, we also need to run backward passes of the differential action model. These
 * calculations are performed by `calcDiff()`. In short, this function builds a linear-quadratic approximation of the
 * differential action model, i.e.,
 * \f[
 * \begin{aligned}
 * &\delta\dot{\mathbf{v}} =
 * \mathbf{f_{q}}\delta\mathbf{q}+\mathbf{f_{v}}\delta\mathbf{v}+\mathbf{f_{u}}\delta\mathbf{u}, &\textrm{(dynamics)}\\
 * &\ell(\delta\mathbf{q},\delta\mathbf{v},\delta\mathbf{u}) = \begin{bmatrix}1 \\ \delta\mathbf{q} \\ \delta\mathbf{v}
 * \\ \delta\mathbf{u}\end{bmatrix}^T \begin{bmatrix}0 & \mathbf{\ell_q}^T & \mathbf{\ell_v}^T & \mathbf{\ell_u}^T \\
 * \mathbf{\ell_q} & \mathbf{\ell_{qq}} &
 * \mathbf{\ell_{qv}} & \mathbf{\ell_{uq}}^T \\
 * \mathbf{\ell_v} & \mathbf{\ell_{vq}} & \mathbf{\ell_{vv}} & \mathbf{\ell_{uv}}^T \\
 * \mathbf{\ell_u} & \mathbf{\ell_{uq}} & \mathbf{\ell_{uv}} & \mathbf{\ell_{uu}}\end{bmatrix} \begin{bmatrix}1 \\
 * \delta\mathbf{q}
 * \\ \delta\mathbf{v} \\
 * \delta\mathbf{u}\end{bmatrix}, &\textrm{(cost)}\\
 * &\mathbf{g_q}\delta\mathbf{q}+\mathbf{g_v}\delta\mathbf{v}+\mathbf{g_u}\delta\mathbf{u}\leq\mathbf{0},
 * &\textrm{(inequality constraints)}\\
 * &\mathbf{h_q}\delta\mathbf{q}+\mathbf{h_v}\delta\mathbf{v}+\mathbf{h_u}\delta\mathbf{u}=\mathbf{0},
 * &\textrm{(equality constraints)} \end{aligned} \f] where
 *  - \f$\mathbf{f_x}=(\mathbf{f_q};\,\, \mathbf{f_v})\in\mathbb{R}^{nv\times ndx}\f$ and
 * \f$\mathbf{f_u}\in\mathbb{R}^{nv\times nu}\f$ are the Jacobians of the dynamics,
 *  - \f$\mathbf{\ell_x}=(\mathbf{\ell_q};\,\, \mathbf{\ell_v})\in\mathbb{R}^{ndx}\f$ and
 * \f$\mathbf{\ell_u}\in\mathbb{R}^{nu}\f$ are the Jacobians of the cost function,
 *  - \f$\mathbf{\ell_{xx}}=(\mathbf{\ell_{qq}}\,\, \mathbf{\ell_{qv}};\,\, \mathbf{\ell_{vq}}\,
 * \mathbf{\ell_{vv}})\in\mathbb{R}^{ndx\times ndx}\f$, \f$\mathbf{\ell_{xu}}=(\mathbf{\ell_q};\,\,
 * \mathbf{\ell_v})\in\mathbb{R}^{ndx\times nu}\f$ and \f$\mathbf{\ell_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the
 * Hessians of the cost function,
 *  - \f$\mathbf{g_x}=(\mathbf{g_q};\,\, \mathbf{g_v})\in\mathbb{R}^{ng\times ndx}\f$ and
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$ are the Jacobians of the inequality constraints, and
 *  - \f$\mathbf{h_x}=(\mathbf{h_q};\,\, \mathbf{h_v})\in\mathbb{R}^{nh\times ndx}\f$ and
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$ are the Jacobians of the equality constraints.
 *
 * Additionally, it is important remark that `calcDiff()` computes the derivatives using the latest stored values by
 * `calc()`. Thus, we need to run first `calc()`.
 *
 * \sa `ActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;

  /**
   * @brief Initialize the differential action model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of control vector
   * @param[in] nr     Dimension of cost-residual vector
   * @param[in] ng     Number of inequality constraints
   * @param[in] nh     Number of equality constraints
   */
  DifferentialActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t nu,
                                     const std::size_t nr = 0, const std::size_t ng = 0, const std::size_t nh = 0);
  virtual ~DifferentialActionModelAbstractTpl();

  /**
   * @brief Compute the system acceleration and cost value
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the total cost value for nodes that depends only on the state
   *
   * It updates the total cost and the system acceleration is not updated as the control input is undefined. This
   * function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost function. It assumes that `calc()` has
   * been run first. This function builds a quadratic approximation of the time-continuous action model (i.e. dynamical
   * system and cost function).
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the derivatives of the cost functions with respect to the state only
   *
   * It updates the derivatives of the cost function with respect to the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the differential action data
   *
   * @return the differential action data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference posture as an equilibrium point, i.e.
   * for \f$\mathbf{f}(\mathbf{q},\mathbf{v}=\mathbf{0},\mathbf{u})=\mathbf{0}\f$
   *
   * @param[in] data    Differential action data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the control
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fx    Jacobian matrix of the dynamics with respect to the state
   * @param[in] A     A matrix to multiply times the Jacobian (`na` x `state_->get_nv()`)
   * @param[out] out  Product between A and the Jacobian of the dynamics with respect to the control (`na` x
   * `state_->get_ndx()`)
   * @param[in] op    Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp = setto) const;

  /**
   * @brief Compute the product between the transpose of the Jacobian of the dynamics with respect to the control and a
   * given matrix A
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fx    Jacobian matrix of the dynamics with respect to the state
   * @param[in] A     A matrix to multiply times the transposed Jacobian (dim `state_->get_nv()` x `na`)
   * @param[out] out  Product between the transposed Jacobian of the dynamics with respect to the control and A
   * (`state_->get_ndx()` x `na`)
   * @param[in] op    Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                                     Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp = setto) const;

  /**
   * @brief Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the control
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fu    Jacobian matrix of the dynamics with respect to the control
   * @param[in] A     A matrix to multiply times the Jacobian (`na` x `state_->get_nv()`)
   * @param[out] out  Product between A and the Jacobian of the dynamics with respect to the control (`na` x `nu_`)
   * @param[in] op    Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp = setto) const;
  /**
   * @brief Compute the product between the transpose of the Jacobian of the dynamics with respect to the control and a
   * given matrix A
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fu    Jacobian matrix of the dynamics with respect to the control
   * @param[in] A     A matrix to multiply times the transposed Jacobian (dim `state_->get_nv()` x `na`)
   * @param[out] out  Product between the transposed Jacobian of the dynamics with respect to the control and A
   * (`nu_` x `na`)
   * @param[in] op    Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                                     Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp = setto) const;

  /**
   * @copybrief quasicStatic()
   *
   * @copydetails quasicStatic()
   *
   * @param[in] data    Differential action data
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   * @return Quasic static commands
   */
  VectorXs quasiStatic_x(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const VectorXs& x,
                         const std::size_t maxiter = 100, const Scalar tol = Scalar(1e-9));

  /**
   * @copybrief multiplyByFx()
   *
   * @copydetails multiplyByFx()
   *
   * @param[in] Fx  Jacobian matrix of the dynamics with respect to the state
   * @param[in] A   A matrix to multiply times the Jacobian (dim `na` x `state_->get_nv()`)
   * @return Product between A and the Jacobian of the dynamics with respect to the control (dim `na` x
   * `state_->get_ndx()`)
   */
  MatrixXs multiplyByFx_A(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A);

  /**
   * @copybrief multiplyFxTransposeBy()
   *
   * @copydetails multiplyFxTransposeBy()
   *
   * @param[in] Fx  Jacobian matrix of the dynamics with respect to the state
   * @param[in] A   A matrix to multiply times the Jacobian (dim `state_->get_nv()` x `na`)
   * @return Product between A and the Jacobian of the dynamics with respect to the state (dim `state_->get_ndx()` x
   * `na`)
   */
  MatrixXsRowMajor multiplyFxTransposeBy_A(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A);

  /**
   * @copybrief multiplyByFu()
   *
   * @copydetails multiplyByFu()
   *
   * @param[in] Fu  Jacobian matrix of the dynamics with respect to the control
   * @param[in] A   A matrix to multiply times the Jacobian (dim `na` x `state_->get_nv()`)
   * @return Product between A and the Jacobian of the dynamics with respect to the control (dim `na` x `nu_`)
   */
  MatrixXs multiplyByFu_A(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A);
  /**
   * @copybrief multiplyFuTransposeBy()
   *
   * @copydetails multiplyFuTransposeBy()
   *
   * @param[in] Fu  Jacobian matrix of the dynamics with respect to the control
   * @param[in] A   A matrix to multiply times the Jacobian (dim `state_->get_nv()` x `na`)
   * @return Product between A and the Jacobian of the dynamics with respect to the state (dim `nu_` x `na`)
   */
  MatrixXsRowMajor multiplyFuTransposeBy_A(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A);

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
  std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  std::size_t get_nh() const;

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
   * @brief Print information on the differential action model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const DifferentialActionModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the differential action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::size_t nu_;                          //!< Control dimension
  std::size_t nr_;                          //!< Dimension of the cost residual
  std::size_t ng_internal_;                 //!< Internal object for storing the number of inequatility constraints
  std::size_t nh_internal_;                 //!< Internal object for storing the number of equatility constraints
  std::size_t* ng_;                         //!< Number of inequality constraints
  std::size_t* nh_;                         //!< Number of equality constraints
  boost::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;                          //!< Neutral state
  VectorXs u_lb_;                           //!< Lower control limits
  VectorXs u_ub_;                           //!< Upper control limits
  bool has_control_limits_;                 //!< Indicates whether any of the control limits is finite

  /**
   * @brief Update the status of the control limits (i.e. if there are defined limits)
   */
  void update_has_control_limits();

  template <class Scalar>
  friend class IntegratedActionModelAbstractTpl;
  template <class Scalar>
  friend class ConstraintModelManagerTpl;
};

template <typename _Scalar>
struct DifferentialActionDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataAbstractTpl(Model<Scalar>* const model)
      : cost(Scalar(0.)),
        xout(model->get_state()->get_nv()),
        Fx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        Fu(model->get_state()->get_nv(), model->get_nu()),
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
    xout.setZero();
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
  virtual ~DifferentialActionDataAbstractTpl() {}

  Scalar cost;    //!< cost value
  VectorXs xout;  //!< evolution state
  MatrixXs Fx;    //!< Jacobian of the dynamics
  MatrixXs Fu;    //!< Jacobian of the dynamics
  VectorXs r;     //!< Cost residual
  VectorXs Lx;    //!< Jacobian of the cost
  VectorXs Lu;    //!< Jacobian of the cost
  MatrixXs Lxx;   //!< Hessian of the cost
  MatrixXs Lxu;   //!< Hessian of the cost
  MatrixXs Luu;   //!< Hessian of the cost
  VectorXs g;     //!< Inequality constraint values
  MatrixXs Gx;    //!< Jacobian of the inequality constraint
  MatrixXs Gu;    //!< Jacobian of the inequality constraint
  VectorXs h;     //!< Equality constraint values
  MatrixXs Hx;    //!< Jacobian of the equality constraint
  MatrixXs Hu;    //!< Jacobian of the equality constraint
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/diff-action-base.hxx"

#endif  // CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

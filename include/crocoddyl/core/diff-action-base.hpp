///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
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
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/core/utils/math.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for differential action model
 *
 * A differential action model is the time-continuous version of an action model, i.e.
 * \f[
 * \begin{aligned}
 * &\dot{\mathbf{v}} = \mathbf{f}(\mathbf{q}, \mathbf{v}, \mathbf{u}), &\textrm{(dynamics)}\\
 * &l(\mathbf{x},\mathbf{u}) = \int_0^{\delta t} a(\mathbf{r}(\mathbf{x},\mathbf{u}))\,dt, &\textrm{(cost)}
 * \end{aligned}
 * \f]
 * where the configuration \f$\mathbf{q}\in\mathcal{Q}\f$ lies in the configuration manifold described with a
 * `nq`-tuple, the velocity \f$\mathbf{v}\in T_{\mathbf{q}}\mathcal{Q}\f$ its a tangent vector to this manifold with
 * `nv` dimension, \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ is the input commands, \f$\mathbf{r}\f$ and \f$a\f$ is a residual
 * and activation functions (see `ActivationModelAbstractTpl`). Both configuration and velocity describe the system
 * space \f$\mathbf{x}\in\mathbf{X}\f$ which lies in the state manifold. Note that the acceleration
 * \f$\dot{\mathbf{v}}\in T_{\mathbf{q}}\mathcal{Q}\f$ lies also in the tangent space of the configuration manifold.
 *
 * The main computations are carrying out in `calc` and `calcDiff`. `calc` computes the acceleration and cost and
 * `calcDiff` computes the derivatives of the dynamics and cost function. Concretely speaking, `calcDiff` builds a
 * linear-quadratic approximation of the differential action, where the dynamics and cost functions have linear and
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
class DifferentialActionModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the differential action model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of control vector
   * @param[in] nr     Dimension of cost-residual vector
   */
  DifferentialActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t nu,
                                     const std::size_t nr = 0);
  virtual ~DifferentialActionModelAbstractTpl();

  /**
   * @brief Compute the system acceleration and cost value
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost function. It assumes that `calc()` has
   * been run first. This function builds a quadratic approximation of the time-continuous action model (i.e. dynamical
   * system and cost function).
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) = 0;

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
   * @copybrief calc()
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point
   */
  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @copybrief calcDiff()
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point
   */
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

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
  boost::shared_ptr<StateAbstract> state_;  //!< Model of the state
  VectorXs unone_;                          //!< Neutral state
  VectorXs u_lb_;                           //!< Lower control limits
  VectorXs u_ub_;                           //!< Upper control limits
  bool has_control_limits_;                 //!< Indicates whether any of the control limits is finite

  void update_has_control_limits();
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
      : cost(0.),
        xout(model->get_state()->get_nv()),
        Fx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        Fu(model->get_state()->get_nv(), model->get_nu()),
        r(model->get_nr()),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()) {
    xout.setZero();
    r.setZero();
    Fx.setZero();
    Fu.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }
  virtual ~DifferentialActionDataAbstractTpl() {}

  Scalar cost;    //!< cost value
  VectorXs xout;  //!< evolution state
  MatrixXs Fx;    //!< Jacobian of the dynamics
  MatrixXs Fu;    //!< Jacobian of the dynamics
  VectorXs r;     //!< Cost residual
  VectorXs Lx;    //!< Jacobian of the cost function
  VectorXs Lu;    //!< Jacobian of the cost function
  MatrixXs Lxx;   //!< Hessian of the cost function
  MatrixXs Lxu;   //!< Hessian of the cost function
  MatrixXs Luu;   //!< Hessian of the cost function
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/diff-action-base.hxx"

#endif  // CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

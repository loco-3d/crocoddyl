///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_STATE_BASE_HPP_
#define CROCODDYL_CORE_STATE_BASE_HPP_

#include <vector>
#include <string>
#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

enum Jcomponent { both = 0, first = 1, second = 2 };
enum AssignmentOp { setto, addto, rmfrom };

inline bool is_a_Jcomponent(Jcomponent firstsecond) {
  return (firstsecond == first || firstsecond == second || firstsecond == both);
}

inline bool is_a_AssignmentOp(AssignmentOp op) { return (op == setto || op == addto || op == rmfrom); }

/**
 * @brief Abstract class for the state representation
 *
 * A state is represented by its operators: difference, integrates, transport and their derivatives.
 * The difference operator returns the value of \f$\mathbf{x}_{1}\ominus\mathbf{x}_{0}\f$ operation.
 * Instead the integrate operator returns the value of \f$\mathbf{x}\oplus\delta\mathbf{x}\f$.
 * These operators are used to compared two points on the state manifold \f$\mathcal{M}\f$ or to advance the state
 * given a tangential velocity (\f$T_\mathbf{x} \mathcal{M}\f$). Therefore the points \f$\mathbf{x}\f$,
 * \f$\mathbf{x}_{0}\f$ and \f$\mathbf{x}_{1}\f$ belong to the manifold \f$\mathcal{M}\f$; and \f$\delta\mathbf{x}\f$
 * or \f$\mathbf{x}_{1}\ominus\mathbf{x}_{0}\f$ lie on its tangential space.
 *
 * \sa `diff()`, `integrate()`, `Jdiff()`, `Jintegrate()` and `JintegrateTransport()`
 */
template <typename _Scalar>
class StateAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the state dimensions
   *
   * @param[in] nx   Dimension of state configuration tuple
   * @param[in] ndx  Dimension of state tangent vector
   */
  StateAbstractTpl(const std::size_t nx, const std::size_t ndx);
  StateAbstractTpl();
  virtual ~StateAbstractTpl();

  /**
   * @brief Generate a zero state
   */
  virtual VectorXs zero() const = 0;

  /**
   * @brief Generate a random state
   */
  virtual VectorXs rand() const = 0;

  /**
   * @brief Compute the state manifold differentiation.
   *
   * The state differentiation is defined as:
   * \f{equation*}{
   *   \delta\mathbf{x} = \mathbf{x}_{1} \ominus \mathbf{x}_{0},
   * \f}
   * where \f$\mathbf{x}_{1}\f$, \f$\mathbf{x}_{0}\f$ are the current and previous state
   * which lie in a manifold \f$\mathcal{M}\f$, and \f$\delta\mathbf{x} \in T_\mathbf{x} \mathcal{M}\f$ is the rate
   * of change in the state in the tangent space of the manifold.
   *
   * @param[in]  x0     Previous state point (size `nx`)
   * @param[in]  x1     Current state point (size `nx`)
   * @param[out] dxout  Difference between the current and previous state points (size `ndx`)
   */
  virtual void diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const = 0;

  /**
   * @brief Compute the state manifold integration.
   *
   * The state integration is defined as:
   * \f{equation*}{
   *   \mathbf{x}_{next} = \mathbf{x} \oplus \delta\mathbf{x},
   * \f}
   * where \f$\mathbf{x}\f$, \f$\mathbf{x}_{next}\f$ are the current and next state
   * which lie in a manifold \f$\mathcal{M}\f$, and \f$\delta\mathbf{x} \in T_\mathbf{x} \mathcal{M}\f$ is the rate
   * of change in the state in the tangent space of the manifold.
   *
   * @param[in]  x     State point (size `nx`)
   * @param[in]  dx    Velocity vector (size `ndx`)
   * @param[out] xout  Next state point (size `nx`)
   */
  virtual void integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const = 0;

  /**
   * @brief Compute the Jacobian of the state manifold differentiation.
   *
   * The state differentiation is defined as:
   * \f{equation*}{
   *   \delta\mathbf{x} = \mathbf{x}_{1} \ominus \mathbf{x}_{0},
   * \f}
   * where \f$\mathbf{x}_{1}\f$, \f$\mathbf{x}_{0}\f$ are the current and previous state
   * which lie in a manifold \f$\mathcal{M}\f$, and \f$\delta\mathbf{x} \in T_\mathbf{x} \mathcal{M}\f$ is the rate
   * of change in the state in the tangent space of the manifold.
   *
   * The Jacobians lie in the tangent space of manifold, i.e. \f$\mathbb{R}^{\textrm{ndx}\times\textrm{ndx}}\f$.
   * Note that the state is represented as a tuple of `nx` values and its dimension is `ndx`.
   * Calling \f$\boldsymbol{\Delta}(\mathbf{x}_{0}, \mathbf{x}_{1}) \f$, the difference function, these Jacobians
   * satisfy the following relationships:
   *  - \f$\boldsymbol{\Delta}(\mathbf{x}_{0},\mathbf{x}_{0}\oplus\delta\mathbf{y}) -
   * \boldsymbol{\Delta}(\mathbf{x}_{0},\mathbf{x}_{1}) = \mathbf{J}_{\mathbf{x}_{1}}\delta\mathbf{y} +
   * \mathbf{o}(\mathbf{x}_{0})\f$.
   *  - \f$\boldsymbol{\Delta}(\mathbf{x}_{0}\oplus\delta\mathbf{y},\mathbf{x}_{1}) -
   * \boldsymbol{\Delta}(\mathbf{x}_{0},\mathbf{x}_{1}) = \mathbf{J}_{\mathbf{x}_{0}}\delta\mathbf{y} +
   * \mathbf{o}(\mathbf{x}_{0})\f$,
   *
   * where \f$\mathbf{J}_{\mathbf{x}_{1}}\f$ and \f$\mathbf{J}_{\mathbf{x}_{0}}\f$ are the Jacobian with respect to the
   * current and previous state, respectively.
   *
   * @param[in] x0           Previous state point (size `nx`)
   * @param[in] x1           Current state point (size `nx`)
   * @param[out] Jfirst      Jacobian of the difference operation relative to the previous state point (size
   * `ndx`\f$\times\f$`ndx`)
   * @param[out] Jsecond     Jacobian of the difference operation relative to the current state point (size
   * `ndx`\f$\times\f$`ndx`)
   * @param[in] firstsecond  Argument (either x0 and / or x1) with respect to which the differentiation is
   * performed.
   */
  virtual void Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const = 0;

  /**
   * @brief Compute the Jacobian of the state manifold integration.
   *
   * The state integration is defined as:
   * \f{equation*}{
   *   \mathbf{x}_{next} = \mathbf{x} \oplus \delta\mathbf{x},
   * \f}
   * where \f$\mathbf{x}\f$, \f$\mathbf{x}_{next}\f$ are the current and next state
   * which lie in a manifold \f$\mathcal{M}\f$, and \f$\delta\mathbf{x} \in T_\mathbf{x} \mathcal{M}\f$ is the rate
   * of change in the state in the tangent space of the manifold.
   *
   * The Jacobians lie in the tangent space of manifold, i.e. \f$\mathbb{R}^{\textrm{ndx}\times\textrm{ndx}}\f$.
   * Note that the state is represented as a tuple of `nx` values and its dimension is `ndx`.
   * Calling \f$ \mathbf{f}(\mathbf{x}, \delta\mathbf{x}) \f$, the integrate function, these Jacobians satisfy the
   * following relationships:
   *  - \f$\mathbf{f}(\mathbf{x}\oplus\delta\mathbf{y},\delta\mathbf{x})\ominus\mathbf{f}(\mathbf{x},\delta\mathbf{x})
   * = \mathbf{J}_\mathbf{x}\delta\mathbf{y} + \mathbf{o}(\delta\mathbf{x})\f$.
   *  - \f$\mathbf{f}(\mathbf{x},\delta\mathbf{x}+\delta\mathbf{y})\ominus\mathbf{f}(\mathbf{x},\delta\mathbf{x}) =
   * \mathbf{J}_{\delta\mathbf{x}}\delta\mathbf{y} + \mathbf{o}(\delta\mathbf{x})\f$,
   *
   * where \f$\mathbf{J}_{\delta\mathbf{x}}\f$ and \f$\mathbf{J}_{\mathbf{x}}\f$ are the Jacobian with respect to the
   * state and velocity, respectively.
   *
   * @param[in] x            State point (size `nx`)
   * @param[in] dx           Velocity vector (size `ndx`)
   * @param[out] Jfirst      Jacobian of the integration operation relative to the state point (size
   * `ndx`\f$\times\f$`ndx`)
   * @param[out] Jsecond     Jacobian of the integration operation relative to the velocity vector (size
   * `ndx`\f$\times\f$`ndx`)
   * @param[in] firstsecond  Argument (either x and / or dx) with respect to which the differentiation is performed
   * @param[in] op           Assignment operator which sets, adds, or removes the given Jacobian matrix
   */
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both, const AssignmentOp op = setto) const = 0;

  /**
   * @brief Parallel transport from x + dx to x.
   *
   * This function performs the parallel transportation of an input matrix whose columns are expressed in the
   * tangent space at \f$\mathbf{x}\oplus\delta\mathbf{x}\f$ to the tangent space at \f$\mathbf{x}\f$ point.
   *
   * @param[in]  x           State point (size `nx`).
   * @param[in]  dx          Velocity vector (size `ndx`)
   * @param[out] Jin         Input matrix (number of rows = `nv`).
   * @param[in] firstsecond  Argument (either x or dx) with respect to which the differentiation of Jintegrate is
   * performed.
   */
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const = 0;

  /**
   * @copybrief diff()
   *
   * @param[in]  x0     Previous state point (size `nx`)
   * @param[in]  x1     Current state point (size `nx`)
   * @return  Difference between the current and previous state points (size `ndx`)
   */
  VectorXs diff_dx(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1);

  /**
   * @copybrief integrate()
   *
   * @param[in]  x     State point (size `nx`)
   * @param[in]  dx    Velocity vector (size `ndx`)
   * @return  Next state point (size `nx`)
   */
  VectorXs integrate_x(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx);

  /**
   * @copybrief jdiff()
   *
   * @param[in]  x0     Previous state point (size `nx`)
   * @param[in]  x1     Current state point (size `nx`)
   * @return  Jacobians
   */
  std::vector<MatrixXs> Jdiff_Js(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                 const Jcomponent firstsecond = both);

  /**
   * @copybrief Jintegrate()
   *
   * @param[in]  x     State point (size `nx`)
   * @param[in]  dx    Velocity vector (size `ndx`)
   * @return  Jacobians
   */
  std::vector<MatrixXs> Jintegrate_Js(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                      const Jcomponent firstsecond = both);

  /**
   * @brief Return the dimension of the state tuple
   */
  std::size_t get_nx() const;

  /**
   * @brief Return the dimension of the tangent space of the state manifold
   */
  std::size_t get_ndx() const;

  /**
   * @brief Return the dimension of the configuration tuple
   */
  std::size_t get_nq() const;

  /**
   * @brief Return the dimension of tangent space of the configuration manifold
   */
  std::size_t get_nv() const;

  /**
   * @brief Return the state lower bound
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the state upper bound
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Indicate if the state has defined limits
   */
  bool get_has_limits() const;

  /**
   * @brief Modify the state lower bound
   */
  void set_lb(const VectorXs& lb);

  /**
   * @brief Modify the state upper bound
   */
  void set_ub(const VectorXs& ub);

 protected:
  void update_has_limits();

  std::size_t nx_;   //!< State dimension
  std::size_t ndx_;  //!< State rate dimension
  std::size_t nq_;   //!< Configuration dimension
  std::size_t nv_;   //!< Velocity dimension
  VectorXs lb_;      //!< Lower state limits
  VectorXs ub_;      //!< Upper state limits
  bool has_limits_;  //!< Indicates whether any of the state limits is finite
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/state-base.hxx"

#endif  // CROCODDYL_CORE_STATE_BASE_HPP_

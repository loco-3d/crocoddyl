///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, INRIA
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

template <typename _Scalar>
class StateAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  StateAbstractTpl(const std::size_t& nx, const std::size_t& ndx);
  StateAbstractTpl();
  virtual ~StateAbstractTpl();

  virtual VectorXs zero() const = 0;
  virtual VectorXs rand() const = 0;
  virtual void diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const = 0;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const = 0;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     Jcomponent firstsecond = both) const = 0;

  /**
   *
   * @brief   Computes the Jacobian of a small variation of the state vector or tangent vector into the tangent space
   * at identity.
   *
   * @details This jacobian has to be interpreted in terms of Lie group, not vector space: as such,
   *          it is expressed in the tangent space only, not the state space.
   *          Calling \f$ f(x, \delta x) \f$ the integrate function, these jacobians satisfy the following
   * relationships in the tangent space:
   *           - Jacobian relative to x: \f$ f(x \oplus \delta y, \delta x) \ominus f(x, \delta x) = J_x \delta y +
   * o(\delta x)\f$.
   *           - Jacobian relative to \delta x: \f$ f(x, \delta x + \delta y) \ominus f(x, \delta x) = J_{\delta x}
   * \delta y + o(\delta x)\f$.
   *
   * @param[in]  x        State Vector.
   * @param[in]  dx       Tangent vector
   * @param[out]  Jfirst   Jacobian of the Integrate operation wrt state vector (size ndx X ndx)
   * @param[out] Jsecond  Jacobian of the Integrate operation wrt tangent vector (size ndx X ndx)
   * @param[in]  arg      Argument (either x or dx) with respect to which the differentiation is performed.
   * @param[in]  op       assignment operator which sets, adds, or removes the jacobian from the matrix given.
   *
   */
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both, const AssignmentOp = setto) const = 0;

  /**
   *
   * @brief   Transport a matrix from the terminal to the originate tangent space of the integrate operation, with
   * respect to the state or the tangent vector arguments.
   *
   * @details This function performs the parallel transportation of an input matrix whose columns are expressed in the
   * tangent space of the integrated element \f$ x \oplus \delta x \f$, to the tangent space at \f$ x \f$. In other
   * words, this functions transforms a tangent vector expressed at \f$ x \oplus \delta x \f$ to a tangent vector
   * expressed at \f$ x \f$, considering that the change of state between \f$ x \oplus \delta x \f$ and \f$ x \f$ may
   * alter the value of this tangent vector. A typical example of parallel transportation is the action operated by a
   * rigid transformation \f$ M \in \text{SE}(3)\f$ on a spatial velocity \f$ v \in \text{se}(3)\f$. In the context of
   * configuration spaces assimilated as vectorial spaces, this operation corresponds to Identity. For Lie groups, its
   * corresponds to the canonical vector field transportation.
   *
   * @param[in]  x        State Vector.
   * @param[in]  dx       Tangent vector
   * @param[out] Jin      Input matrix (number of rows = model.nv).
   * @param[in]  arg      Argument (either x or dx) with respect to which the differentiation of Jintegrate is
   * performed.
   *
   */
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const = 0;
  VectorXs diff_dx(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1);
  VectorXs integrate_x(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx);
  std::vector<MatrixXs> Jdiff_Js(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                 Jcomponent firstsecond = both);
  std::vector<MatrixXs> Jintegrate_Js(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                      const Jcomponent firstsecond = both, const AssignmentOp op = setto);

  const std::size_t& get_nx() const;
  const std::size_t& get_ndx() const;
  const std::size_t& get_nq() const;
  const std::size_t& get_nv() const;

  const VectorXs& get_lb() const;
  const VectorXs& get_ub() const;
  bool const& get_has_limits() const;

  void set_lb(const VectorXs& lb);
  void set_ub(const VectorXs& ub);

  void update_has_limits();

 protected:
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

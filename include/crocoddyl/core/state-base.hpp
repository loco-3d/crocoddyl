///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
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
enum AssignmentOp { setto = 0, addto = 1, rmfrom = 2 };

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
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          Jcomponent firstsecond = both) const = 0;
  virtual void JintegrateOp(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                            Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                            const Jcomponent firstsecond, const AssignmentOp) const = 0;  
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const = 0;
  VectorXs diff_dx(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1);
  VectorXs integrate_x(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx);
  std::vector<MatrixXs> Jdiff_Js(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                 Jcomponent firstsecond = both);
  std::vector<MatrixXs> Jintegrate_Js(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                      Jcomponent firstsecond = both);

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

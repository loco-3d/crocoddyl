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

#ifdef PYTHON_BINDINGS

 public:
  VectorXs diff_wrap(const VectorXs& x0, const VectorXs& x1) {
    VectorXs dxout = VectorXs::Zero(ndx_);
    diff(x0, x1, dxout);
    return dxout;
  }
  VectorXs integrate_wrap(const VectorXs& x, const VectorXs& dx) {
    VectorXs xout = VectorXs::Zero(nx_);
    integrate(x, dx, xout);
    return xout;
  }
  std::vector<MatrixXs> Jdiff_wrap(const VectorXs& x0, const VectorXs& x1, Jcomponent firstsecond = both) {
    MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<MatrixXs> Jacs;
    Jdiff(x0, x1, Jfirst, Jsecond, firstsecond);
    switch (firstsecond) {
      case both:
        Jacs.push_back(Jfirst);
        Jacs.push_back(Jsecond);
        break;
      case first:
        Jacs.push_back(Jfirst);
        break;
      case second:
        Jacs.push_back(Jsecond);
        break;
      default:
        Jacs.push_back(Jfirst);
        Jacs.push_back(Jsecond);
        break;
    }
    return Jacs;
  }
  std::vector<MatrixXs> Jintegrate_wrap(const VectorXs& x, const VectorXs& dx, Jcomponent firstsecond = both) {
    MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<MatrixXs> Jacs;
    Jintegrate(x, dx, Jfirst, Jsecond, firstsecond);
    switch (firstsecond) {
      case both:
        Jacs.push_back(Jfirst);
        Jacs.push_back(Jsecond);
        break;
      case first:
        Jacs.push_back(Jfirst);
        break;
      case second:
        Jacs.push_back(Jsecond);
        break;
      default:
        Jacs.push_back(Jfirst);
        Jacs.push_back(Jsecond);
        break;
    }
    return Jacs;
  }
#endif
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/state-base.hxx"

#endif  // CROCODDYL_CORE_STATE_BASE_HPP_

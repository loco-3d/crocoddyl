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
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;

  StateAbstractTpl(const std::size_t& nx, const std::size_t& ndx);
  StateAbstractTpl();
  virtual ~StateAbstractTpl();

  virtual typename MathBase::VectorXs zero() const = 0;
  virtual typename MathBase::VectorXs rand() const = 0;
  virtual void diff(const Eigen::Ref<const typename MathBase::VectorXs>& x0,
                    const Eigen::Ref<const typename MathBase::VectorXs>& x1,
                    Eigen::Ref<typename MathBase::VectorXs> dxout) const = 0;
  virtual void integrate(const Eigen::Ref<const typename MathBase::VectorXs>& x,
                         const Eigen::Ref<const typename MathBase::VectorXs>& dx,
                         Eigen::Ref<typename MathBase::VectorXs> xout) const = 0;
  virtual void Jdiff(const Eigen::Ref<const typename MathBase::VectorXs>& x0,
                     const Eigen::Ref<const typename MathBase::VectorXs>& x1,
                     Eigen::Ref<typename MathBase::MatrixXs> Jfirst, Eigen::Ref<typename MathBase::MatrixXs> Jsecond,
                     Jcomponent firstsecond = both) const = 0;
  virtual void Jintegrate(const Eigen::Ref<const typename MathBase::VectorXs>& x,
                          const Eigen::Ref<const typename MathBase::VectorXs>& dx,
                          Eigen::Ref<typename MathBase::MatrixXs> Jfirst,
                          Eigen::Ref<typename MathBase::MatrixXs> Jsecond, Jcomponent firstsecond = both) const = 0;

  const std::size_t& get_nx() const;
  const std::size_t& get_ndx() const;
  const std::size_t& get_nq() const;
  const std::size_t& get_nv() const;

  const typename MathBase::VectorXs& get_lb() const;
  const typename MathBase::VectorXs& get_ub() const;
  bool const& get_has_limits() const;

  void set_lb(const typename MathBase::VectorXs& lb);
  void set_ub(const typename MathBase::VectorXs& ub);

  void update_has_limits();

 protected:
  std::size_t nx_;                  //!< State dimension
  std::size_t ndx_;                 //!< State rate dimension
  std::size_t nq_;                  //!< Configuration dimension
  std::size_t nv_;                  //!< Velocity dimension
  typename MathBase::VectorXs lb_;  //!< Lower state limits
  typename MathBase::VectorXs ub_;  //!< Upper state limits
  bool has_limits_;                 //!< Indicates whether any of the state limits is finite

#ifdef PYTHON_BINDINGS

 public:
  typename MathBase::VectorXs diff_wrap(const typename MathBase::VectorXs& x0, const typename MathBase::VectorXs& x1) {
    typename MathBase::VectorXs dxout = MathBase::VectorXs::Zero(ndx_);
    diff(x0, x1, dxout);
    return dxout;
  }
  typename MathBase::VectorXs integrate_wrap(const typename MathBase::VectorXs& x,
                                             const typename MathBase::VectorXs& dx) {
    typename MathBase::VectorXs xout = MathBase::VectorXs::Zero(nx_);
    integrate(x, dx, xout);
    return xout;
  }
  std::vector<typename MathBase::MatrixXs> Jdiff_wrap(const typename MathBase::VectorXs& x0,
                                                      const typename MathBase::VectorXs& x1,
                                                      std::string firstsecond = "both") {
    typename MathBase::MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<typename MathBase::MatrixXs> Jacs;
    if (firstsecond == "both") {
      Jdiff(x0, x1, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    } else if (firstsecond == "first") {
      Jdiff(x0, x1, Jfirst, Jsecond, first);
      Jacs.push_back(Jfirst);
    } else if (firstsecond == "second") {
      Jdiff(x0, x1, Jfirst, Jsecond, second);
      Jacs.push_back(Jsecond);
    } else {
      Jdiff(x0, x1, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    }
    return Jacs;
  }
  std::vector<typename MathBase::MatrixXs> Jintegrate_wrap(const typename MathBase::VectorXs& x,
                                                           const typename MathBase::VectorXs& dx,
                                                           std::string firstsecond = "both") {
    typename MathBase::MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<typename MathBase::MatrixXs> Jacs;
    if (firstsecond == "both") {
      Jintegrate(x, dx, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    } else if (firstsecond == "first") {
      Jintegrate(x, dx, Jfirst, Jsecond, first);
      Jacs.push_back(Jfirst);
    } else if (firstsecond == "second") {
      Jintegrate(x, dx, Jfirst, Jsecond, second);
      Jacs.push_back(Jsecond);
    } else {
      Jintegrate(x, dx, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
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

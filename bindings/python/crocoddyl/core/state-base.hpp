
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

#include <string>

#include "crocoddyl/core/state-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class StateAbstractTpl_wrap : public StateAbstractTpl<Scalar>,
                              public bp::wrapper<StateAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(StateBase, StateAbstractTpl_wrap)

  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename State::VectorXs VectorXs;
  typedef typename State::MatrixXs MatrixXs;
  using State::lb_;
  using State::ndx_;
  using State::nq_;
  using State::nv_;
  using State::nx_;
  using State::ub_;

  StateAbstractTpl_wrap(std::size_t nx, std::size_t ndx)
      : State(nx, ndx), bp::wrapper<State>() {
    enableMultithreading() = false;
  }
  explicit StateAbstractTpl_wrap() {}

  VectorXs zero() const override {
    return bp::call<VectorXs>(this->get_override("zero").ptr());
  }

  VectorXs rand() const override {
    return bp::call<VectorXs>(this->get_override("rand").ptr());
  }

  VectorXs diff_wrap(const Eigen::Ref<const VectorXs>& x0,
                     const Eigen::Ref<const VectorXs>& x1) const {
    if (static_cast<std::size_t>(x0.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x1 has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    return bp::call<VectorXs>(this->get_override("diff").ptr(), (VectorXs)x0,
                              (VectorXs)x1);
  }

  void diff(const Eigen::Ref<const VectorXs>& x0,
            const Eigen::Ref<const VectorXs>& x1,
            Eigen::Ref<VectorXs> dxout) const override {
    dxout = diff_wrap(x0, x1);
  }

  VectorXs integrate_wrap(const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& dx) const {
    if (static_cast<std::size_t>(x.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dx.size()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "dx has wrong dimension (it should be " +
                                      std::to_string(ndx_) + ")");
    }
    return bp::call<VectorXs>(this->get_override("integrate").ptr(),
                              (VectorXs)x, (VectorXs)dx);
  }

  void integrate(const Eigen::Ref<const VectorXs>& x,
                 const Eigen::Ref<const VectorXs>& dx,
                 Eigen::Ref<VectorXs> x1out) const override {
    x1out = integrate_wrap(x, dx);
  }

  void Jdiff(const Eigen::Ref<const VectorXs>& x0,
             const Eigen::Ref<const VectorXs>& x1, Eigen::Ref<MatrixXs> Jfirst,
             Eigen::Ref<MatrixXs> Jsecond,
             const Jcomponent firstsecond) const override {
    bp::list res = Jdiff_wrap(x0, x1, firstsecond);
    switch (firstsecond) {
      case first: {
        Jfirst.derived() = bp::extract<MatrixXs>(res[0])();
        break;
      }
      case second: {
        Jsecond.derived() = bp::extract<MatrixXs>(res[0])();
        break;
      }
      case both: {
        Jfirst.derived() = bp::extract<MatrixXs>(res[0])();
        Jsecond.derived() = bp::extract<MatrixXs>(res[1])();
        break;
      }
      default: {
        Jfirst.derived() = bp::extract<MatrixXs>(res[0])();
        Jsecond.derived() = bp::extract<MatrixXs>(res[1])();
        break;
      }
    }
  }

  bp::list Jdiff_wrap(const Eigen::Ref<const VectorXs>& x0,
                      const Eigen::Ref<const VectorXs>& x1,
                      const Jcomponent firstsecond) const {
    assert_pretty(
        is_a_Jcomponent(firstsecond),
        ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x1 has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }

    bp::list Jacs;
    switch (firstsecond) {
      case first: {
        MatrixXs J =
            bp::call<MatrixXs>(this->get_override("Jdiff").ptr(), (VectorXs)x0,
                               (VectorXs)x1, firstsecond);
        Jacs.append(J);
        break;
      }
      case second: {
        MatrixXs J =
            bp::call<MatrixXs>(this->get_override("Jdiff").ptr(), (VectorXs)x0,
                               (VectorXs)x1, firstsecond);
        Jacs.append(J);
        break;
      }
      case both: {
        Jacs = bp::call<bp::list>(this->get_override("Jdiff").ptr(),
                                  (VectorXs)x0, (VectorXs)x1, firstsecond);
        break;
      }
      default: {
        Jacs = bp::call<bp::list>(this->get_override("Jdiff").ptr(),
                                  (VectorXs)x0, (VectorXs)x1, firstsecond);
        break;
      }
    }
    return Jacs;
  }

  void Jintegrate(const Eigen::Ref<const VectorXs>& x,
                  const Eigen::Ref<const VectorXs>& dx,
                  Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                  const Jcomponent firstsecond,
                  const AssignmentOp op) const override {
    bp::list res = Jintegrate_wrap(x, dx, firstsecond);
    if (firstsecond == first || firstsecond == both) {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
          static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " +
                            std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      switch (op) {
        case setto: {
          Jfirst.derived() = bp::extract<MatrixXs>(res[0])();
          break;
        }
        case addto: {
          Jfirst.derived() += bp::extract<MatrixXs>(res[0])();
          break;
        }
        case rmfrom: {
          Jfirst.derived() -= bp::extract<MatrixXs>(res[0])();
          break;
        }
        default: {
          throw_pretty(
              "Invalid argument: allowed operators: setto, addto, rmfrom");
          break;
        }
      }
    }
    if (firstsecond == second || firstsecond == both) {
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
          static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " +
                            std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      switch (op) {
        case setto: {
          Jsecond.derived() = bp::extract<MatrixXs>(res[0])();
          break;
        }
        case addto: {
          Jsecond.derived() += bp::extract<MatrixXs>(res[0])();
          break;
        }
        case rmfrom: {
          Jsecond.derived() -= bp::extract<MatrixXs>(res[0])();
          break;
        }
        default: {
          throw_pretty(
              "Invalid argument: allowed operators: setto, addto, rmfrom");
          break;
        }
      }
    }
  }

  bp::list Jintegrate_wrap(const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& dx,
                           const Jcomponent firstsecond) const {
    assert_pretty(
        is_a_Jcomponent(firstsecond),
        ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dx.size()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "dx has wrong dimension (it should be " +
                                      std::to_string(ndx_) + ")");
    }

    bp::list Jacs;
    switch (firstsecond) {
      case first: {
        MatrixXs J = bp::call<MatrixXs>(this->get_override("Jintegrate").ptr(),
                                        (VectorXs)x, (VectorXs)dx, firstsecond);
        Jacs.append(J);
        break;
      }
      case second: {
        MatrixXs J = bp::call<MatrixXs>(this->get_override("Jintegrate").ptr(),
                                        (VectorXs)x, (VectorXs)dx, firstsecond);
        Jacs.append(J);
        break;
      }
      case both: {
        Jacs = bp::call<bp::list>(this->get_override("Jintegrate").ptr(),
                                  (VectorXs)x, (VectorXs)dx, firstsecond);
        break;
      }
      default: {
        Jacs = bp::call<bp::list>(this->get_override("Jintegrate").ptr(),
                                  (VectorXs)x, (VectorXs)dx, firstsecond);
        break;
      }
    }
    return Jacs;
  }

  void JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& dx,
                           Eigen::Ref<MatrixXs> Jin,
                           const Jcomponent firstsecond) const override {
    Jin = JintegrateTransport_wrap(x, dx, Jin, firstsecond);
  }

  MatrixXs JintegrateTransport_wrap(const Eigen::Ref<const VectorXs>& x,
                                    const Eigen::Ref<const VectorXs>& dx,
                                    Eigen::Ref<MatrixXs> Jin,
                                    const Jcomponent firstsecond) const {
    assert_pretty(
        is_a_Jcomponent(firstsecond),
        ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x.size()) != nx_) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dx.size()) != ndx_) {
      throw_pretty(
          "Invalid argument: " << "dx has wrong dimension (it should be " +
                                      std::to_string(ndx_) + ")");
    }
    return bp::call<MatrixXs>(this->get_override("JintegrateTransport").ptr(),
                              (VectorXs)x, (VectorXs)dx, (MatrixXs)Jin,
                              firstsecond);
  }

  template <typename NewScalar>
  StateAbstractTpl_wrap<NewScalar> cast() const {
    typedef StateAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(nx_, ndx_);
    return ret;
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jdiffs, StateAbstract::Jdiff_Js, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jintegrates,
                                       StateAbstract::Jintegrate_Js, 2, 3)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

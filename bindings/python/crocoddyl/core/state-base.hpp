
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

#include <string>
#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class StateAbstract_wrap : public StateAbstract, public bp::wrapper<StateAbstract> {
 public:
  using StateAbstract::lb_;
  using StateAbstract::ub_;

  StateAbstract_wrap(int nx, int ndx) : StateAbstract(nx, ndx), bp::wrapper<StateAbstract>() {}

  Eigen::VectorXd zero() const { return bp::call<Eigen::VectorXd>(this->get_override("zero").ptr()); }

  Eigen::VectorXd rand() const { return bp::call<Eigen::VectorXd>(this->get_override("rand").ptr()); }

  Eigen::VectorXd diff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0,
                            const Eigen::Ref<const Eigen::VectorXd>& x1) const {
    if (static_cast<std::size_t>(x0.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    return bp::call<Eigen::VectorXd>(this->get_override("diff").ptr(), (Eigen::VectorXd)x0, (Eigen::VectorXd)x1);
  }

  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) const {
    dxout = diff_wrap(x0, x1);
  }

  Eigen::VectorXd integrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& dx) const {
    if (static_cast<std::size_t>(x.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dx.size()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }
    return bp::call<Eigen::VectorXd>(this->get_override("integrate").ptr(), (Eigen::VectorXd)x, (Eigen::VectorXd)dx);
  }

  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> x1out) const {
    x1out = integrate_wrap(x, dx);
  }

  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent firstsecond) const {
    bp::list res = Jdiff_wrap(x0, x1, firstsecond);
    switch (firstsecond) {
      case first: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        break;
      }
      case second: {
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        break;
      }
      case both: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
        break;
      }
      default: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
      }
    }
  }

  bp::list Jdiff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                      Jcomponent firstsecond) const {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    bp::list Jacs;
    switch (firstsecond) {
      case first: {
        Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0,
                                                      (Eigen::VectorXd)x1, firstsecond);
        Jacs.append(J);
        break;
      }
      case second: {
        Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0,
                                                      (Eigen::VectorXd)x1, firstsecond);
        Jacs.append(J);
        break;
      }
      case both: {
        Jacs = bp::call<bp::list>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0, (Eigen::VectorXd)x1,
                                  firstsecond);
        break;
      }
      default: {
        Jacs = bp::call<bp::list>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0, (Eigen::VectorXd)x1,
                                  firstsecond);
      }
    }
    return Jacs;
  }

  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                  const Jcomponent firstsecond, const AssignmentOp op) const {
    bp::list res = Jintegrate_wrap(x, dx, firstsecond, op);
    switch (firstsecond) {
      case first: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        break;
      }
      case second: {
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        break;
      }
      case both: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
        break;
      }
      default: {
        Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
        Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
      }
    }
  }

  bp::list Jintegrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                           const Jcomponent firstsecond, const AssignmentOp op) const {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dx.size()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }

    bp::list Jacs;
    switch (firstsecond) {
      case first: {
        Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x,
                                                      (Eigen::VectorXd)dx, firstsecond, op);
        Jacs.append(J);
        break;
      }
      case second: {
        Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x,
                                                      (Eigen::VectorXd)dx, firstsecond, op);
        Jacs.append(J);
        break;
      }
      case both: {
        Jacs = bp::call<bp::list>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x, (Eigen::VectorXd)dx,
                                  firstsecond, op);
        break;
      }
      default: {
        Jacs = bp::call<bp::list>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x, (Eigen::VectorXd)dx,
                                  firstsecond, op);
      }
    }
    return Jacs;
  }

  void JintegrateTransport(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                           Eigen::Ref<Eigen::MatrixXd> Jin, const Jcomponent _firstsecond) const {}
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jdiffs, StateAbstract::Jdiff_Js, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jintegrates, StateAbstract::Jintegrate_Js, 2, 4)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PYTHON_CORE_STATE_BASE_HPP_
#define CROCODDYL_PYTHON_CORE_STATE_BASE_HPP_

#include <crocoddyl/core/state-base.hpp>
#include <iostream>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class StateAbstract_wrap : public StateAbstract,
                           public bp::wrapper<StateAbstract> {
 public:
  using StateAbstract::nx;
  using StateAbstract::ndx;

  StateAbstract_wrap(int nx, int ndx) : StateAbstract(nx, ndx), bp::wrapper<StateAbstract>() {}

  Eigen::VectorXd zero() { return bp::call<Eigen::VectorXd>(this->get_override("zero").ptr()); }

  Eigen::VectorXd rand() { return bp::call<Eigen::VectorXd>(this->get_override("rand").ptr()); }

  Eigen::VectorXd diff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0,
                            const Eigen::Ref<const Eigen::VectorXd>& x1) {
    return bp::call<Eigen::VectorXd>(this->get_override("diff").ptr(),
                                     (Eigen::VectorXd) x0,
                                     (Eigen::VectorXd) x1);
  }

  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0,
            const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) {
    dxout = diff_wrap(x0, x1);
  }

  Eigen::VectorXd integrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& dx) {
    return bp::call<Eigen::VectorXd>(this->get_override("integrate").ptr(),
                                     (Eigen::VectorXd) x,
                                     (Eigen::VectorXd) dx);
  }

  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                 const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> x1out) {
    x1out = integrate_wrap(x, dx);
  }

  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0,
             const Eigen::Ref<const Eigen::VectorXd>& x1,
             Eigen::Ref<Eigen::MatrixXd> Jfirst,
             Eigen::Ref<Eigen::MatrixXd> Jsecond,
             Jcomponent _firstsecond) {
    std::string firstsecond;
    switch (_firstsecond) {
    case first: {
      firstsecond = "first";
      break;
    } case second: {
      firstsecond = "second";
      break;
    } case both: {
      firstsecond = "both";
      break;
    } default: {
      firstsecond = "both";
    }}

    bp::list res = Jdiff_wrap(x0, x1, firstsecond);
    if (firstsecond == "both") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
    } else if (firstsecond == "first") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    } else if (firstsecond == "second") {
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    }
  }

  bp::list Jdiff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0,
                      const Eigen::Ref<const Eigen::VectorXd>& x1,
                      std::string firstsecond) {
    assert(firstsecond == "both" || firstsecond == "first" || firstsecond == "second");
    if (firstsecond == "both") {
      bp::list Jacs = bp::call<bp::list>(this->get_override("Jdiff").ptr(),
                                         (Eigen::VectorXd) x0,
                                         (Eigen::VectorXd) x1,
                                         firstsecond);
      return Jacs;
    } else {
      Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jdiff").ptr(),
                                                    (Eigen::VectorXd) x0,
                                                    (Eigen::VectorXd) x1,
                                                    firstsecond);
      bp::list list;
      list.append(J);
      return list;
    }
  }

  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x0,
                  const Eigen::Ref<const Eigen::VectorXd>& x1,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst,
                  Eigen::Ref<Eigen::MatrixXd> Jsecond,
                  Jcomponent _firstsecond) {
    std::string firstsecond;
    switch (_firstsecond) {
    case first: {
      firstsecond = "first";
      break;
    } case second: {
      firstsecond = "second";
      break;
    } case both: {
      firstsecond = "both";
      break;
    } default: {
      firstsecond = "both";
    }}

    bp::list res = Jintegrate_wrap(x0, x1, firstsecond);
    if (firstsecond == "both") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
    } else if (firstsecond == "first") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    } else if (firstsecond == "second") {
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    }
  }

  bp::list Jintegrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0,
                           const Eigen::Ref<const Eigen::VectorXd>& x1,
                           std::string firstsecond) {
    assert(firstsecond == "both" || firstsecond == "first" || firstsecond == "second");
    if (firstsecond == "both") {
      bp::list Jacs =
          bp::call<bp::list>(this->get_override("Jintegrate").ptr(),
                             (Eigen::VectorXd) x0,
                             (Eigen::VectorXd) x1,
                             firstsecond);
      return Jacs;
    } else {
      Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jintegrate").ptr(),
                                                  (Eigen::VectorXd) x0,
                                                  (Eigen::VectorXd) x1,
                                                  firstsecond);
      bp::list list;
      list.append(J);
      return list;
    }
  }
};

void exposeStateAbstract() {
  bp::class_<StateAbstract_wrap, boost::noncopyable>(
      "StateAbstract",
      R"(Abstract class for the state representation.

        A state is represented by its operators: difference, integrates and their derivatives.
        The difference operator returns the value of x1 [-] x2 operation. Instead the integrate
        operator returns the value of x [+] dx. These operators are used to compared two points
        on the state manifold M or to advance the state given a tangential velocity (Tx M).
        Therefore the points x, x1 and x2 belongs to the manifold M; and dx or x1 [-] x2 lie
        on its tangential space.)",
      bp::init<int, int>(bp::args(" self", " nx", " ndx"),
                         R"(Initialize the state dimensions.

:param nx: dimension of state configuration vector,
:param ndx: dimension of state tangent vector)"))
      .def("zero", pure_virtual(&StateAbstract_wrap::zero), bp::args(" self"),
           R"(Return a zero reference state.

:return zero reference state)")
      .def("rand", pure_virtual(&StateAbstract_wrap::rand), bp::args(" self"),
           R"(Return a random reference state.

:return random reference state)")
      .def("diff", pure_virtual(&StateAbstract_wrap::diff_wrap),
           bp::args(" self", " x0", " x1"),
           R"(Operator that differentiates the two state points.

It returns the value of x1 [-] x0 operation. Note tha x0 and x1 are points in the state
manifold (in M). Instead the operator result lies in the tangent-space of M.
:param x0: current state (dim state.nx).
:param x1: next state (dim state.nx).
:return x1 [-] x0 value (dim state.ndx).)")
      .def("integrate", pure_virtual(&StateAbstract_wrap::integrate),
           bp::args(" self", " x", " dx"),
           R"(Operator that integrates the current state.

It returns the value of x [+] dx operation. x and dx are points in the statstate.diff(x0,x1)d (in M)
and its tangent, respectively. Note that the operator result lies on M too.state.diff(x0,x1)
:param x: current state (dim state.nx).
:param dx: displacement of the state (dim state.ndx).
:return x [+] dx value (dim state.nx).)")
      .def("Jdiff", pure_virtual(&StateAbstract_wrap::Jdiff_wrap),
           bp::args(" self", " x0", " x1", " firstsecond = 'both'"),
           R"(Compute the partial derivatives of difference operator.

For a given state, the difference operator (x1 [-] x0) is defined by diff(x0, x1). Instead
here it is described its partial derivatives, i.e. \partial{diff(x0, x1)}{x0} and
\partial{diff(x0, x1)}{x1}. By default, this function returns the derivatives of the
first and second argument (i.e. firstsecond='both'). However we ask for a specific partial
derivative by setting firstsecond='first' or firstsecond='second'.
:param x0: current state (dim state.nx).
:param x1: next state (dim state.nx).
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the diff(x0, x1) function)")
      .def("Jintegrate", pure_virtual(&StateAbstract_wrap::Jintegrate_wrap),
           bp::args(" self", " x", " dx", " firstsecond = 'both'"),
           R"(Compute the partial derivatives of integrate operator.

For a given state, the integrate operator (x [+] dx) is defined by integrate(x, dx).
Instead here it is described its partial derivatives, i.e. \partial{integrate(x, dx)}{x}
and \partial{integrate(x, dx)}{dx}. By default, this function returns the derivatives of
the first and second argument (i.e. firstsecond='both'). However we ask for a specific
partial derivative by setting firstsecond='first' or firstsecond='second'.
:param x: current state (dim state.nx).
:param dx: displacement of the state (dim state.ndx).
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the integrate(x, dx) function)")
      .add_property("nx", &StateAbstract_wrap::nx, "dimension of state configuration vector")
      .add_property("ndx", &StateAbstract_wrap::ndx, "dimension of state tangent vector");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // CROCODDYL_PYTHON_CORE_STATE_BASE_HPP_

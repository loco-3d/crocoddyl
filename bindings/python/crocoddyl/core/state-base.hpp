///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

#include <string>
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class StateAbstract_wrap : public StateAbstract, public bp::wrapper<StateAbstract> {
 public:
  StateAbstract_wrap(int nx, int ndx) : StateAbstract(nx, ndx), bp::wrapper<StateAbstract>() {}

  Eigen::VectorXd zero() { return bp::call<Eigen::VectorXd>(this->get_override("zero").ptr()); }

  Eigen::VectorXd rand() { return bp::call<Eigen::VectorXd>(this->get_override("rand").ptr()); }

  Eigen::VectorXd diff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1) {
    assert(static_cast<std::size_t>(x0.size()) == nx_ && "x0 has wrong dimension");
    assert(static_cast<std::size_t>(x1.size()) == nx_ && "x1 has wrong dimension");
    return bp::call<Eigen::VectorXd>(this->get_override("diff").ptr(), (Eigen::VectorXd)x0, (Eigen::VectorXd)x1);
  }

  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) {
    dxout = diff_wrap(x0, x1);
  }

  Eigen::VectorXd integrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& dx) {
    assert(static_cast<std::size_t>(x.size()) == nx_ && "x has wrong dimension");
    assert(static_cast<std::size_t>(dx.size()) == ndx_ && "dx has wrong dimension");
    return bp::call<Eigen::VectorXd>(this->get_override("integrate").ptr(), (Eigen::VectorXd)x, (Eigen::VectorXd)dx);
  }

  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> x1out) {
    x1out = integrate_wrap(x, dx);
  }

  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent _firstsecond) {
    std::string firstsecond;
    switch (_firstsecond) {
      case first: {
        firstsecond = "first";
        break;
      }
      case second: {
        firstsecond = "second";
        break;
      }
      case both: {
        firstsecond = "both";
        break;
      }
      default: { firstsecond = "both"; }
    }

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

  bp::list Jdiff_wrap(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                      std::string firstsecond) {
    assert((firstsecond == "both" || firstsecond == "first" || firstsecond == "second") &&
           "firstsecond must be one of the Jcomponent {both, first, second}");
    assert(static_cast<std::size_t>(x0.size()) == nx_ && "x0 has wrong dimension");
    assert(static_cast<std::size_t>(x1.size()) == nx_ && "x1 has wrong dimension");

    if (firstsecond == "both") {
      bp::list Jacs =
          bp::call<bp::list>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0, (Eigen::VectorXd)x1, firstsecond);
      return Jacs;
    } else {
      Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jdiff").ptr(), (Eigen::VectorXd)x0,
                                                    (Eigen::VectorXd)x1, firstsecond);
      bp::list list;
      list.append(J);
      return list;
    }
  }

  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent _firstsecond) {
    std::string firstsecond;
    switch (_firstsecond) {
      case first: {
        firstsecond = "first";
        break;
      }
      case second: {
        firstsecond = "second";
        break;
      }
      case both: {
        firstsecond = "both";
        break;
      }
      default: { firstsecond = "both"; }
    }

    bp::list res = Jintegrate_wrap(x, dx, firstsecond);
    if (firstsecond == "both") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[1])();
    } else if (firstsecond == "first") {
      Jfirst.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    } else if (firstsecond == "second") {
      Jsecond.derived() = bp::extract<Eigen::MatrixXd>(res[0])();
    }
  }

  bp::list Jintegrate_wrap(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                           std::string firstsecond) {
    assert((firstsecond == "both" || firstsecond == "first" || firstsecond == "second") &&
           "firstsecond must be one of the Jcomponent {both, first, second}");
    assert(static_cast<std::size_t>(x.size()) == nx_ && "x has wrong dimension");
    assert(static_cast<std::size_t>(dx.size()) == ndx_ && "dx has wrong dimension");

    if (firstsecond == "both") {
      bp::list Jacs = bp::call<bp::list>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x,
                                         (Eigen::VectorXd)dx, firstsecond);
      return Jacs;
    } else {
      Eigen::MatrixXd J = bp::call<Eigen::MatrixXd>(this->get_override("Jintegrate").ptr(), (Eigen::VectorXd)x,
                                                    (Eigen::VectorXd)dx, firstsecond);
      bp::list list;
      list.append(J);
      return list;
    }
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jdiffs, StateAbstract::Jdiff_wrap, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Jintegrates, StateAbstract::Jintegrate_wrap, 2, 3)

void exposeStateAbstract() {
  bp::class_<StateAbstract_wrap, boost::noncopyable>(
      "StateAbstract",
      "Abstract class for the state representation.\n\n"
      "A state is represented by its operators: difference, integrates and their derivatives.\n"
      "The difference operator returns the value of x1 [-] x2 operation. Instead the integrate\n"
      "operator returns the value of x [+] dx. These operators are used to compared two points\n"
      "on the state manifold M or to advance the state given a tangential velocity (Tx M).\n"
      "Therefore the points x, x1 and x2 belongs to the manifold M; and dx or x1 [-] x2 lie\n"
      "on its tangential space.",
      bp::init<int, int>(bp::args(" self", " nx", " ndx"),
                         "Initialize the state dimensions.\n\n"
                         ":param nx: dimension of state configuration vector\n"
                         ":param ndx: dimension of state tangent vector"))
      .def("zero", pure_virtual(&StateAbstract_wrap::zero), bp::args(" self"),
           "Return a zero reference state.\n\n"
           ":return zero reference state")
      .def("rand", pure_virtual(&StateAbstract_wrap::rand), bp::args(" self"),
           "Return a random reference state.\n\n"
           ":return random reference state")
      .def("diff", pure_virtual(&StateAbstract_wrap::diff_wrap), bp::args(" self", " x0", " x1"),
           "Operator that differentiates the two state points.\n\n"
           "It returns the value of x1 [-] x0 operation. Note tha x0 and x1 are points in the state\n"
           "manifold (in M). Instead the operator result lies in the tangent-space of M.\n"
           ":param x0: current state (dim state.nx).\n"
           ":param x1: next state (dim state.nx).\n"
           ":return x1 [-] x0 value (dim state.ndx).")
      .def("integrate", pure_virtual(&StateAbstract_wrap::integrate), bp::args(" self", " x", " dx"),
           "Operator that integrates the current state.\n\n"
           "It returns the value of x [+] dx operation. x and dx are points in the state.diff(x0,x1) (in M)\n"
           "and its tangent, respectively. Note that the operator result lies on M too.\n"
           ":param x: current state (dim state.nx).\n"
           ":param dx: displacement of the state (dim state.ndx).\n"
           ":return x [+] dx value (dim state.nx).")
      .def("Jdiff", pure_virtual(&StateAbstract_wrap::Jdiff_wrap),
           bp::args(" self", " x0", " x1", " firstsecond = 'both'"),
           "Compute the partial derivatives of difference operator.\n\n"
           "The difference operator (x1 [-] x0) is defined by diff(x0, x1). Instead Jdiff\n"
           "computes its partial derivatives, i.e. \\partial{diff(x0, x1)}{x0} and\n"
           "\\partial{diff(x0, x1)}{x1}. By default, this function returns the derivatives of the\n"
           "first and second argument (i.e. firstsecond='both'). However we can also specific the\n"
           "partial derivative for the first and second variables by setting firstsecond='first'\n"
           "or firstsecond='second', respectively.\n"
           ":param x0: current state (dim state.nx).\n"
           ":param x1: next state (dim state.nx).\n"
           ":param firstsecond: desired partial derivative\n"
           ":return the partial derivative(s) of the diff(x0, x1) function")
      .def("Jintegrate", pure_virtual(&StateAbstract_wrap::Jintegrate_wrap),
           bp::args(" self", " x", " dx", " firstsecond = 'both'"),
           "Compute the partial derivatives of integrate operator.\n\n"
           "The integrate operator (x [+] dx) is defined by integrate(x, dx). Instead Jintegrate\n"
           "computes its partial derivatives, i.e. \\partial{integrate(x, dx)}{x} and\n"
           "\\partial{integrate(x, dx)}{dx}. By default, this function returns the derivatives of\n"
           "the first and second argument (i.e. firstsecond='both'). However we ask for a specific\n"
           "partial derivative by setting firstsecond='first' or firstsecond='second'.\n"
           ":param x: current state (dim state.nx).\n"
           ":param dx: displacement of the state (dim state.ndx).\n"
           ":param firstsecond: desired partial derivative\n"
           ":return the partial derivative(s) of the integrate(x, dx) function")
      .add_property("nx",
                    bp::make_function(&StateAbstract_wrap::get_nx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of state configuration vector")
      .add_property("ndx",
                    bp::make_function(&StateAbstract_wrap::get_ndx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of state tangent vector")
      .add_property("nq",
                    bp::make_function(&StateAbstract_wrap::get_nq, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of configuration vector")
      .add_property("nv",
                    bp::make_function(&StateAbstract_wrap::get_nv, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of configuration tangent vector");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_STATE_BASE_HPP_

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_

#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeStateMultibody() {
  bp::class_<StateMultibody, bp::bases<StateAbstract> >(
      "StateMultibody",
      "Multibody state defined using Pinocchio.\n\n"
      "Pinocchio defines operators for integrating or differentiating the robot's\n"
      "configuration space. And here we assume that the state is defined by the\n"
      "robot's configuration and its joint velocities (x=[q,v]). Generally speaking,\n"
      "q lies on the manifold configuration manifold (M) and v in its tangent space\n"
      "(Tx M). Additionally the Pinocchio allows us to compute analytically the\n"
      "Jacobians for the differentiate and integrate operators. Note that this code\n"
      "can be reused in any robot that is described through its Pinocchio model.",
      bp::init<pinocchio::Model*>(
          bp::args(" self", " pinocchioModel"),
          "Initialize the multibody state given a Pinocchio model.\n\n"
          ":param pinocchioModel: pinocchio model (i.e. multibody model)")[bp::with_custodian_and_ward<1, 2>()])
      .def("zero", &StateMultibody::zero, bp::args(" self"),
           "Return the neutral robot configuration with zero velocity.\n\n"
           ":return neutral robot configuration with zero velocity")
      .def("rand", &StateMultibody::rand, bp::args(" self"),
           "Return a random reference state.\n\n"
           ":return random reference state")
      .def("diff", &StateMultibody::diff_wrap, bp::args(" self", " x0", " x1"),
           "Operator that differentiates the two robot states.\n\n"
           "It returns the value of x1 [-] x0 operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           ":param x0: current state (dim state.nx()).\n"
           ":param x1: next state (dim state.nx()).\n"
           ":return x1 - x0 value (dim state.nx()).")
      .def("integrate", &StateMultibody::integrate_wrap, bp::args(" self", " x", " dx"),
           "Operator that integrates the current robot state.\n\n"
           "It returns the value of x [+] dx operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           "Futhermore there is no timestep here (i.e. dx = v*dt), note this if you're\n"
           "integrating a velocity v during an interval dt.\n"
           ":param x: current state (dim state.nx()).\n"
           ":param dx: displacement of the state (dim state.ndx()).\n"
           ":return x + dx value (dim state.nx()).")
      .def("Jdiff", &StateMultibody::Jdiff_wrap,
           Jdiffs(bp::args(" self", " x0", " x1", " firstsecond = 'both'"),
                  "Compute the partial derivatives of the diff operator.\n\n"
                  "Both Jacobian matrices are represented throught an identity matrix, with the exception\n"
                  "that the robot's root is defined as free-flying joint (SE(3)). By default, this\n"
                  "function returns the derivatives of the first and second argument (i.e.\n"
                  "firstsecond='both'). However we ask for a specific partial derivative by setting\n"
                  "firstsecond='first' or firstsecond='second'.\n"
                  ":param x0: current state (dim state.nx()).\n"
                  ":param x1: next state (dim state.nx()).\n"
                  ":param firstsecond: desired partial derivative\n"
                  ":return the partial derivative(s) of the diff(x0, x1) function"))
      .def("Jintegrate", &StateMultibody::Jintegrate_wrap,
           Jintegrates(bp::args(" self", " x", " dx", " firstsecond = 'both'"),
                       "Compute the partial derivatives of arithmetic addition.\n\n"
                       "Both Jacobian matrices are represented throught an identity matrix. with the exception\n"
                       "that the robot's root is defined as free-flying joint (SE(3)). By default, this\n"
                       "function returns the derivatives of the first and second argument (i.e.\n"
                       "firstsecond='both'). However we ask for a specific partial derivative by setting\n"
                       "firstsecond='first' or firstsecond='second'.\n"
                       ":param x: current state (dim state.nx()).\n"
                       ":param dx: displacement of the state (dim state.ndx()).\n"
                       ":param firstsecond: desired partial derivative\n"
                       ":return the partial derivative(s) of the integrate(x, dx) function"))
      .add_property("model",
                    bp::make_function(&StateMultibody::get_model,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "pinocchio model");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
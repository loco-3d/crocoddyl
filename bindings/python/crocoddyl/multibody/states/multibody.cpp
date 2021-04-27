///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeStateMultibody() {
  bp::register_ptr_to_python<boost::shared_ptr<crocoddyl::StateMultibody> >();

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
      bp::init<boost::shared_ptr<pinocchio::Model> >(
          bp::args("self", "pinocchioModel"),
          "Initialize the multibody state given a Pinocchio model.\n\n"
          ":param pinocchioModel: pinocchio model (i.e. multibody model)")[bp::with_custodian_and_ward<1, 2>()])
      .def("zero", &StateMultibody::zero, bp::args("self"),
           "Return the neutral robot configuration with zero velocity.\n\n"
           ":return neutral robot configuration with zero velocity")
      .def("rand", &StateMultibody::rand, bp::args("self"),
           "Return a random reference state.\n\n"
           ":return random reference state")
      .def("diff", &StateMultibody::diff_dx, bp::args("self", "x0", "x1"),
           "Operator that differentiates the two robot states.\n\n"
           "It returns the value of x1 [-] x0 operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           ":param x0: current state (dim state.nx()).\n"
           ":param x1: next state (dim state.nx()).\n"
           ":return x1 - x0 value (dim state.nx()).")
      .def("integrate", &StateMultibody::integrate_x, bp::args("self", "x", "dx"),
           "Operator that integrates the current robot state.\n\n"
           "It returns the value of x [+] dx operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           "Futhermore there is no timestep here (i.e. dx = v*dt), note this if you're\n"
           "integrating a velocity v during an interval dt.\n"
           ":param x: current state (dim state.nx()).\n"
           ":param dx: displacement of the state (dim state.ndx()).\n"
           ":return x + dx value (dim state.nx()).")
      .def("Jdiff", &StateMultibody::Jdiff_Js,
           Jdiffs(bp::args("self", "x0", "x1", "firstsecond"),
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
      .def("Jintegrate", &StateMultibody::Jintegrate_Js,
           Jintegrates(bp::args("self", "x", "dx", "firstsecond"),
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
      .add_property("pinocchio",
                    bp::make_function(&StateMultibody::get_pinocchio, bp::return_value_policy<bp::return_by_value>()),
                    "pinocchio model");
}

}  // namespace python
}  // namespace crocoddyl

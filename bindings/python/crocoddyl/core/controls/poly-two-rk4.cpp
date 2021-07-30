///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/controls/poly-two-rk4.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyTwoRK4() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlParametrizationModelPolyTwoRK4> >();

  bp::class_<ControlParametrizationModelPolyTwoRK4, bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyTwoRK4",
      "Quadratic control.\n\n"
      "This control is a quadratic function of time (normalized in [0,1])."
      "The first third of the parameter vector contains the initial value of the differential control w, "
      "the second third contains the value of w at t=0.5, and the last third is the final value of w at time t=1.",
      bp::init<std::size_t>(bp::args("self", "nw"),
                            "Initialize the control dimensions.\n\n"
                            ":param nw: dimension of differential control space\n"))
      .def<void (ControlParametrizationModelPolyTwoRK4::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>("calc", &ControlParametrizationModelPolyTwoRK4::calc,
                                                           bp::args("self", "data", "t", "u"),
                                                           "Compute the control value.\n\n"
                                                           ":param data: the data on which the method operates.\n"
                                                           ":param t: normalized time in [0, 1].\n"
                                                           ":param u: control parameters (dim control.nu).")
      .def<void (ControlParametrizationModelPolyTwoRK4::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>("params", &ControlParametrizationModelPolyTwoRK4::params,
                                                           bp::args("self", "data", "t", "w"),
                                                           "Compute the control parameters.\n\n"
                                                           ":param data: the data on which the method operates.\n"
                                                           ":param t: normalized time in [0, 1].\n"
                                                           ":param w: control value (dim control.nw).")
      .def("convertBounds", &ControlParametrizationModelPolyTwoRK4::convertBounds, bp::args("self", "w_lb", "w_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param w_lb: lower bounds on u (dim control.nw).\n"
           ":param w_ub: upper bounds on u (dim control.nw).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.nu).")
      .def<void (ControlParametrizationModelPolyTwoRK4::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyTwoRK4::calcDiff, bp::args("self", "data", "t", "u"),
          "Compute the Jacobian of the control value with respect to the control parameters.\n"
          "It assumes that calc has been run first.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param u: control parameters (dim control.nu).")
      .def("multiplyByJacobian", &ControlParametrizationModelPolyTwoRK4::multiplyByJacobian_J,
           bp::args("self", "t", "u", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).\n"
           ":param A: matrix to multiply (dim na x control.nw).\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.nu).")
      .def(
          "multiplyJacobianTransposeBy", &ControlParametrizationModelPolyTwoRK4::multiplyJacobianTransposeBy_J,
          bp::args("self", "t", "u", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          ":param t: normalized time in [0, 1].\n"
          ":param u: control parameters (dim control.nu).\n"
          ":param A: matrix to multiply (dim control.nw x na).\n"
          ":return Product between the partial derivative of the value function (transposed) and A (dim control.nu x "
          "na).");
}

}  // namespace python
}  // namespace crocoddyl

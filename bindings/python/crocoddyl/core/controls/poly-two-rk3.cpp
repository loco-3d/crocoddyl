///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/controls/poly-two-rk3.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyTwoRK3() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlParametrizationModelPolyTwoRK3> >();

  bp::class_<ControlParametrizationModelPolyTwoRK3, bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyTwoRK3",
      "Second-order polynomial control for RK3 integrator.\n\n"
      "This control is a quadratic function of time (normalized in [0,1])."
      "The first third of the parameter vector contains the initial value of the differential control w, "
      "the second third contains the value of w at t=1/3, and the last third contains its value at t=2/3.",
      bp::init<std::size_t>(bp::args("self", "nw"),
                            "Initialize the control dimensions.\n\n"
                            ":param nw: dimension of differential control space"))
      .def<void (ControlParametrizationModelPolyTwoRK3::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>("calc", &ControlParametrizationModelPolyTwoRK3::calc,
                                                           bp::args("self", "data", "t", "u"),
                                                           "Compute the control value.\n\n"
                                                           ":param data: control-parametrization data\n"
                                                           ":param t: normalized time in [0, 1]\n"
                                                           ":param u: control parameters (dim control.nu)")
      .def<void (ControlParametrizationModelPolyTwoRK3::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyTwoRK3::calcDiff, bp::args("self", "data", "t", "u"),
          "Compute the value of the Jacobian of the control with respect to the parameters.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param u: control parameters (dim control.nu)")
      .def<void (ControlParametrizationModelPolyTwoRK3::*)(
          const boost::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>("params", &ControlParametrizationModelPolyTwoRK3::params,
                                                           bp::args("self", "data", "t", "w"),
                                                           "Compute the control parameters.\n\n"
                                                           ":param data: control-parametrization data\n"
                                                           ":param t: normalized time in [0, 1]\n"
                                                           ":param w: control value (dim control.nw)")
      .def("convertBounds", &ControlParametrizationModelPolyTwoRK3::convertBounds, bp::args("self", "w_lb", "w_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param w_lb: lower bounds on u (dim control.nw)\n"
           ":param w_ub: upper bounds on u (dim control.nw)\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.nu)")
      .def("multiplyByJacobian", &ControlParametrizationModelPolyTwoRK3::multiplyByJacobian_J,
           bp::args("self", "data", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: control-parametrization data\n"
           ":param A: matrix to multiply (dim na x control.nw)\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.nu)")
      .def(
          "multiplyJacobianTransposeBy", &ControlParametrizationModelPolyTwoRK3::multiplyJacobianTransposeBy_J,
          bp::args("self", "data", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: control-parametrization data\n"
          ":param A: matrix to multiply (dim control.nw x na)\n"
          ":return Product between the partial derivative of the value function (transposed) and A (dim control.nu x "
          "na)");

  boost::python::register_ptr_to_python<boost::shared_ptr<ControlParametrizationDataPolyTwoRK3> >();

  bp::class_<ControlParametrizationDataPolyTwoRK3, bp::bases<ControlParametrizationDataAbstract> >(
      "ControlParametrizationDataPolyTwoRK3", "Control-parametrization data for the second-order polynomial control.",
      bp::init<ControlParametrizationModelPolyTwoRK3*>(bp::args("self", "model"),
                                                       "Create control-parametrization data.\n\n"
                                                       ":param model: second-order polynomial control model"))
      .add_property("c", bp::make_getter(&ControlParametrizationDataPolyTwoRK3::c, bp::return_internal_reference<>()),
                    "polynomial coefficients of the second-order control model");
}

}  // namespace python
}  // namespace crocoddyl

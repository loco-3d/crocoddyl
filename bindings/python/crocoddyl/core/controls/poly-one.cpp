///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/controls/poly-one.hpp"

#include "python/crocoddyl/core/control-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyOne() {
  bp::register_ptr_to_python<
      std::shared_ptr<ControlParametrizationModelPolyOne> >();

  bp::class_<ControlParametrizationModelPolyOne,
             bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyOne",
      "Linear polynomial control.\n\n"
      "This control is a linear function of time (normalized in [0,1])."
      "The first half of the parameter vector contains the initial value of "
      "the differential control w, "
      "whereas the second half contains the value of w at t=0.5.",
      bp::init<std::size_t>(
          bp::args("self", "nw"),
          "Initialize the control dimensions.\n\n"
          ":param nw: dimension of differential control space"))
      .def<void (ControlParametrizationModelPolyOne::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calc", &ControlParametrizationModelPolyOne::calc,
          bp::args("self", "data", "t", "u"),
          "Compute the control value.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param u: control parameters (dim control.nu)")
      .def<void (ControlParametrizationModelPolyOne::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyOne::calcDiff,
          bp::args("self", "data", "t", "u"),
          "Compute the Jacobian of the control value with respect to the "
          "control parameters.\n"
          "It assumes that calc has been run first.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param u: control parameters (dim control.nu)")
      .def("createData", &ControlParametrizationModelPolyOne::createData,
           bp::args("self"), "Create the poly-one data.")
      .def<void (ControlParametrizationModelPolyOne::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "params", &ControlParametrizationModelPolyOne::params,
          bp::args("self", "data", "t", "w"),
          "Compute the control parameters.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param w: control value (dim control.nw)")
      .def("convertBounds", &ControlParametrizationModelPolyOne::convertBounds,
           bp::args("self", "w_lb", "w_ub"),
           "Convert the bounds on the control to bounds on the control "
           "parameters.\n\n"
           ":param w_lb: lower bounds on u (dim control.nw).\n"
           ":param w_ub: upper bounds on u (dim control.nw).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control "
           "parameters (dim control.nu).")
      .def("multiplyByJacobian",
           &ControlParametrizationModelPolyOne::multiplyByJacobian_J,
           ControlParametrizationModelAbstract_multiplyByJacobian_J_wrap(
               bp::args("self", "data", "A", "op"),
               "Compute the product between the given matrix A and the "
               "derivative "
               "of the control with respect to the "
               "parameters.\n\n"
               "It assumes that calc has been run first.\n"
               ":param data: control-parametrization data\n"
               ":param A: matrix to multiply (dim na x control.nw)\n"
               ":op assignment operator which sets, adds, or removes the given "
               "results\n"
               ":return Product between A and the partial derivative of the "
               "value "
               "function (dim na x control.nu)"))
      .def(
          "multiplyJacobianTransposeBy",
          &ControlParametrizationModelPolyOne::multiplyJacobianTransposeBy_J,
          ControlParametrizationModelAbstract_multiplyJacobianTransposeBy_J_wrap(
              bp::args("self", "data", "A", "op"),
              "Compute the product between the transpose of the derivative of "
              "the "
              "control with respect to the parameters\n"
              "and a given matrix A.\n\n"
              "It assumes that calc has been run first.\n"
              ":param data: control-parametrization data\n"
              ":param A: matrix to multiply (dim control.nw x na)\n"
              ":op assignment operator which sets, adds, or removes the given "
              "results\n"
              ":return Product between the partial derivative of the value "
              "function (transposed) and A (dim control.nu x "
              "na)"))
      .def(CopyableVisitor<ControlParametrizationModelPolyOne>());

  boost::python::register_ptr_to_python<
      std::shared_ptr<ControlParametrizationDataPolyOne> >();

  bp::class_<ControlParametrizationDataPolyOne,
             bp::bases<ControlParametrizationDataAbstract> >(
      "ControlParametrizationDataPolyOne",
      "Control-parametrization data for the linear polynomial control.",
      bp::init<ControlParametrizationModelPolyOne*>(
          bp::args("self", "model"),
          "Create control-parametrization data.\n\n"
          ":param model: linear polynomial control model"))
      .add_property("c",
                    bp::make_getter(&ControlParametrizationDataPolyOne::c,
                                    bp::return_internal_reference<>()),
                    "polynomial coefficients of the linear control model that "
                    "depends on time")
      .def(CopyableVisitor<ControlParametrizationDataPolyOne>());
}

}  // namespace python
}  // namespace crocoddyl

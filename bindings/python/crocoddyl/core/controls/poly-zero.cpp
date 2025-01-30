///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/controls/poly-zero.hpp"

#include "python/crocoddyl/core/control-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyZero() {
  bp::register_ptr_to_python<
      std::shared_ptr<ControlParametrizationModelPolyZero> >();

  bp::class_<ControlParametrizationModelPolyZero,
             bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyZero",
      "Constant control.\n\n"
      "This control is a constant."
      "The parameter vector corresponds to the constant value of the "
      "differential control w, that is w=p.",
      bp::init<std::size_t>(
          bp::args("self", "nw"),
          "Initialize the control dimensions.\n\n"
          ":param nw: dimension of differential control space"))
      .def<void (ControlParametrizationModelPolyZero::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calc", &ControlParametrizationModelPolyZero::calc,
          bp::args("self", "data", "t", "u"),
          "Compute the control value.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param u: control parameters (dim control.nu)")
      .def<void (ControlParametrizationModelPolyZero::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyZero::calcDiff,
          bp::args("self", "data", "t", "u"),
          "Compute the Jacobian of the control value with respect to the "
          "control parameters.\n"
          "It assumes that calc has been run first.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param u: control parameters (dim control.nu)")
      .def("createData", &ControlParametrizationModelPolyZero::createData,
           bp::args("self"), "Create the poly-zero data.")
      .def<void (ControlParametrizationModelPolyZero::*)(
          const std::shared_ptr<ControlParametrizationDataAbstract>&, double,
          const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "params", &ControlParametrizationModelPolyZero::params,
          bp::args("self", "data", "t", "u"),
          "Compute the control parameters.\n\n"
          ":param data: control-parametrization data\n"
          ":param t: normalized time in [0, 1]\n"
          ":param w: control value (dim control.nw)")
      .def("convertBounds", &ControlParametrizationModelPolyZero::convertBounds,
           bp::args("self", "u_lb", "u_ub"),
           "Convert the bounds on the control to bounds on the control "
           "parameters.\n\n"
           ":param w_lb: lower bounds on w (dim control.nw)\n"
           ":param w_ub: upper bounds on w (dim control.nw)\n"
           ":return p_lb, p_ub: lower and upper bounds on the control "
           "parameters (dim control.nu)")
      .def("multiplyByJacobian",
           &ControlParametrizationModelPolyZero::multiplyByJacobian_J,
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
          &ControlParametrizationModelPolyZero::multiplyJacobianTransposeBy_J,
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
      .def(CopyableVisitor<ControlParametrizationModelPolyZero>());
}

}  // namespace python
}  // namespace crocoddyl

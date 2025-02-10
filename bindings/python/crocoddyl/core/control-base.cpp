///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, LAAS-CNRS, University of Edinburgh,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise controld in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/control-base.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationAbstract() {
  // Register custom converters between std::vector and Python list
  typedef std::shared_ptr<ControlParametrizationModelAbstract>
      ControlParametrizationModelPtr;
  typedef std::shared_ptr<ControlParametrizationDataAbstract>
      ControlParametrizationDataPtr;
  StdVectorPythonVisitor<std::vector<ControlParametrizationModelPtr>,
                         true>::expose("StdVec_ControlParametrizationModel");
  StdVectorPythonVisitor<std::vector<ControlParametrizationDataPtr>,
                         true>::expose("StdVec_ControlParametrizationData");

  bp::register_ptr_to_python<
      std::shared_ptr<ControlParametrizationModelAbstract> >();

  bp::class_<ControlParametrizationModelAbstract_wrap, boost::noncopyable>(
      "ControlParametrizationModelAbstract",
      "Abstract class for the control parametrization.\n\n"
      "A control is a function of time (normalized in [0,1]) and the control "
      "parameters u.",
      bp::init<std::size_t, std::size_t>(
          bp::args("self", "nw", "nu"),
          "Initialize the control dimensions.\n\n"
          ":param nw: dimension of control inputs\n"
          ":param nu: dimension of control parameters"))
      .def("calc",
           pure_virtual(&ControlParametrizationModelAbstract_wrap::calc),
           bp::args("self", "t", "u"),
           "Compute the control inputs.\n\n"
           ":param data: control-parametrization data\n"
           ":param t: normalized time in [0, 1]\n"
           ":param u: control parameters (dim control.nu)")
      .def("calcDiff",
           pure_virtual(&ControlParametrizationModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "t", "u"),
           "Compute the Jacobian of the control inputs with respect to the "
           "control parameters.\n"
           "It assumes that calc has been run first.\n\n"
           ":param data: control-parametrization data\n"
           ":param t: normalized time in [0, 1]\n"
           ":param u: control parameters (dim control.nu)")
      .def("createData", &ControlParametrizationModelAbstract_wrap::createData,
           &ControlParametrizationModelAbstract_wrap::default_createData,
           bp::args("self"),
           "Create the control-parametrization data.\n\n"
           "Each control parametrization model has its own data that needs to "
           "be allocated.\n"
           "This function returns the allocated data for a predefined control "
           "parametrization model.\n"
           ":return data.")
      .def("params",
           pure_virtual(&ControlParametrizationModelAbstract_wrap::params),
           bp::args("self", "data", "t", "w"),
           "Update the control parameters u for a specified time t given the "
           "control input w.\n\n"
           ":param data: control-parametrization data\n"
           ":param t: normalized time in [0, 1]\n"
           ":param w: control inputs (dim control.nw)")
      .def("convertBounds",
           pure_virtual(
               &ControlParametrizationModelAbstract_wrap::convertBounds_wrap),
           bp::args("self", "w_lb", "w_ub"),
           "Convert the bounds on the control inputs w to bounds on the "
           "control parameters u.\n\n"
           ":param w_lb: control lower bounds (dim control.nw)\n"
           ":param w_ub: control upper bounds (dim control.nw)\n"
           ":return p_lb, p_ub: lower and upper bounds on the control "
           "parameters (dim control.nu)")
      .def("multiplyByJacobian",
           pure_virtual(&ControlParametrizationModelAbstract_wrap::
                            multiplyByJacobian_wrap),
           bp::args("self", "data", "A"),
           "Compute the product between the given matrix A and the derivative "
           "of the control input \n"
           "with respect to the control parameters (i.e., A*dw_du).\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: control-parametrization data\n"
           ":param A: matrix to multiply (dim na x control.nw)\n"
           ":return Product between A and the partial derivative of the calc "
           "function (dim na x control.nu)")
      .def("multiplyJacobianTransposeBy",
           pure_virtual(&ControlParametrizationModelAbstract_wrap::
                            multiplyJacobianTransposeBy_wrap),
           bp::args("self", "data", "A"),
           "Compute the product between the transpose of the derivative of the "
           "control input \n"
           "with respect to the control parameters and a given matrix A (i.e., "
           "dw_du^T*A).\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: control-parametrization data\n"
           ":param A: matrix to multiply (dim control.nw x na)\n"
           ":return Product between the partial derivative of the calc "
           "function (transposed) and A (dim control.nu x "
           "na)")
      .add_property(
          "nw",
          bp::make_function(&ControlParametrizationModelAbstract_wrap::get_nw),
          "dimension of control inputs")
      .add_property(
          "nu",
          bp::make_function(&ControlParametrizationModelAbstract_wrap::get_nu),
          "dimension of the control parameters");

  bp::register_ptr_to_python<
      std::shared_ptr<ControlParametrizationDataAbstract> >();

  bp::class_<ControlParametrizationDataAbstract>(
      "ControlParametrizationDataAbstract",
      "Abstract class for control parametrization data.\n",
      bp::init<ControlParametrizationModelAbstract*>(
          bp::args("self", "model"),
          "Create common data shared between control parametrization "
          "models.\n\n"
          ":param model: control parametrization model"))
      .add_property("w",
                    bp::make_getter(&ControlParametrizationDataAbstract::w,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ControlParametrizationDataAbstract::w),
                    "differential control")
      .add_property("u",
                    bp::make_getter(&ControlParametrizationDataAbstract::u,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ControlParametrizationDataAbstract::u),
                    "control parameters")
      .add_property(
          "dw_du",
          bp::make_getter(&ControlParametrizationDataAbstract::dw_du,
                          bp::return_internal_reference<>()),
          bp::make_setter(&ControlParametrizationDataAbstract::dw_du),
          "Jacobian of the differential control wrt the control parameters")
      .def(CopyableVisitor<ControlParametrizationDataAbstract>());
}

}  // namespace python
}  // namespace crocoddyl

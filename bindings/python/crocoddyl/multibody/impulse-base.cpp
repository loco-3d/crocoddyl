///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/impulse-base.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeImpulseAbstract() {
  bp::register_ptr_to_python<std::shared_ptr<ImpulseModelAbstract> >();

  bp::class_<ImpulseModelAbstract_wrap, boost::noncopyable>(
      "ImpulseModelAbstract",
      "Abstract impulse model.\n\n"
      "It defines a template for impulse models.\n"
      "The calc and calcDiff functions compute the impulse Jacobian\n"
      "the derivatives respectively.",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::ReferenceFrame,
               std::size_t>(bp::args("self", "state", "type", "nc"),
                            "Initialize the impulse model.\n\n"
                            ":param state: state of the multibody system\n"
                            ":param type: type of impulse\n"
                            ":param nc: dimension of impulse model"))
      .def(bp::init<std::shared_ptr<StateMultibody>, std::size_t>(
          bp::args("self", "state", "nc"),
          "Initialize the impulse model.\n\n"
          ":param state: state of the multibody system\n"
          ":param nc: dimension of impulse model"))
      .def("calc", pure_virtual(&ImpulseModelAbstract_wrap::calc),
           bp::args("self", "data", "x"),
           "Compute the impulse Jacobian\n"
           ":param data: impulse data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", pure_virtual(&ImpulseModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "x"),
           "Compute the derivatives of impulse Jacobian\n"
           "It assumes that calc has been run first.\n"
           ":param data: impulse data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateForce", pure_virtual(&ImpulseModelAbstract_wrap::updateForce),
           bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: impulse data\n"
           ":param force: force vector (dimension nc)")
      .def("updateForceDiff", &ImpulseModelAbstract_wrap::updateForceDiff,
           bp::args("self", "data", "df_dx"),
           "Update the Jacobian of the impulse force.\n\n"
           ":param data: impulse data\n"
           ":param df_dx: Jacobian of the impulse force (dimension nc*ndx)")
      .def("setZeroForce", &ImpulseModelAbstract_wrap::setZeroForce,
           bp::args("self", "data"),
           "Set zero the spatial force.\n\n"
           ":param data: contact data")
      .def("setZeroForceDiff", &ImpulseModelAbstract_wrap::setZeroForceDiff,
           bp::args("self", "data"),
           "Set zero the derivatives of the spatial force.\n\n"
           ":param data: contact data")
      .def("createData", &ImpulseModelAbstract_wrap::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined impulse.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .def("createData", &ImpulseModelAbstract_wrap::default_createData,
           bp::with_custodian_and_ward_postcall<0, 2>())
      .add_property(
          "state",
          bp::make_function(&ImpulseModelAbstract_wrap::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property("ni",
                    bp::make_function(&ImpulseModelAbstract_wrap::get_nc,
                                      deprecated<>("Deprecated. Use nc")),
                    "dimension of impulse")
      .add_property("nc", bp::make_function(&ImpulseModelAbstract_wrap::get_nc),
                    "dimension of impulse")
      .add_property("id", &ImpulseModelAbstract_wrap::get_id,
                    &ImpulseModelAbstract_wrap::set_id, "reference frame id")
      .add_property("type",
                    bp::make_function(&ImpulseModelAbstract_wrap::get_type),
                    &ImpulseModelAbstract_wrap::set_type, "type of impulse")
      .def(CopyableVisitor<ImpulseModelAbstract_wrap>())
      .def(PrintableVisitor<ImpulseModelAbstract>());

  bp::register_ptr_to_python<std::shared_ptr<ImpulseDataAbstract> >();

  bp::class_<ImpulseDataAbstract, bp::bases<ForceDataAbstract> >(
      "ImpulseDataAbstract", "Abstract class for impulse data.\n\n",
      bp::init<ImpulseModelAbstract*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create common data shared between impulse models.\n\n"
          ":param model: impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 3>()])
      .add_property("fXj",
                    bp::make_getter(&ImpulseDataAbstract::fXj,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataAbstract::fXj),
                    "action matrix from contact to local frames")
      .add_property("dv0_dq",
                    bp::make_getter(&ImpulseDataAbstract::dv0_dq,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataAbstract::dv0_dq),
                    "Jacobian of the previous impulse velocity")
      .add_property("dtau_dq",
                    bp::make_getter(&ImpulseDataAbstract::dtau_dq,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataAbstract::dtau_dq),
                    "force contribution to dtau_dq")
      .def(CopyableVisitor<ImpulseDataAbstract>());
}

}  // namespace python
}  // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactWrenchCone() {
  bp::register_ptr_to_python<
      std::shared_ptr<ResidualModelContactWrenchCone> >();

  bp::class_<ResidualModelContactWrenchCone, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactWrenchCone",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               WrenchCone, std::size_t, bp::optional<bool> >(
          bp::args("self", "state", "id", "fref", "nu", "fwddyn"),
          "Initialize the contact wrench cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: contact wrench cone\n"
          ":param nu: dimension of control vector\n"
          ":param fwddyn: indicate if we have a forward dynamics problem "
          "(True) or inverse dynamics problem (False) "
          "(default True)"))
      .def(bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    WrenchCone>(
          bp::args("self", "state", "id", "fref"),
          "Initialize the contact wrench cone residual model.\n\n"
          "The default nu is obtained from state.nv. Note that this "
          "constructor can be used for forward-dynamics\n"
          "cases only.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: contact wrench cone"))
      .def<void (ResidualModelContactWrenchCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactWrenchCone::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the contact wrench cone residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelContactWrenchCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactWrenchCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactWrenchCone::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact wrench cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelContactWrenchCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactWrenchCone::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact wrench cone residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the contact wrench cone residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property(
          "id", bp::make_function(&ResidualModelContactWrenchCone::get_id),
          bp::make_function(
              &ResidualModelContactWrenchCone::set_id,
              deprecated<>(
                  "Deprecated. Do not use set_id, instead create a new model")),
          "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelContactWrenchCone::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelContactWrenchCone::set_reference,
          "reference contact wrench cone")
      .def(CopyableVisitor<ResidualModelContactWrenchCone>());

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataContactWrenchCone> >();

  bp::class_<ResidualDataContactWrenchCone, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactWrenchCone",
      "Data for contact wrench cone residual.\n\n",
      bp::init<ResidualModelContactWrenchCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact wrench cone residual data.\n\n"
          ":param model: contact wrench cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataContactWrenchCone::contact,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactWrenchCone::contact),
          "contact data associated with the current residual")
      .def(CopyableVisitor<ResidualDataContactWrenchCone>());
}

}  // namespace python
}  // namespace crocoddyl

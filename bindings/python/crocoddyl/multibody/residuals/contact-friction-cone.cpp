///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactFrictionCone() {
  bp::register_ptr_to_python<
      std::shared_ptr<ResidualModelContactFrictionCone> >();

  bp::class_<ResidualModelContactFrictionCone,
             bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactFrictionCone",
      "This residual function is defined as r = A*f, where A, f describe the "
      "linearized friction cone and\n"
      "the spatial force, respectively.",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               FrictionCone, std::size_t, bp::optional<bool> >(
          bp::args("self", "state", "id", "fref", "nu", "fwddyn"),
          "Initialize the contact friction cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: frame friction cone\n"
          ":param nu: dimension of control vector\n"
          ":param fwddyn: indicate if we have a forward dynamics problem "
          "(True) or inverse "
          "dynamics problem (False) (default True)"))
      .def(bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    FrictionCone>(
          bp::args("self", "state", "id", "fref"),
          "Initialize the contact friction cone residual model.\n\n"
          "The default nu is obtained from state.nv. Note that this "
          "constructor can be used for forward-dynamics\n"
          "cases only.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: frame friction cone"))
      .def<void (ResidualModelContactFrictionCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactFrictionCone::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the contact friction cone residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelContactFrictionCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactFrictionCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactFrictionCone::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact friction cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelContactFrictionCone::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def(
          "createData", &ResidualModelContactFrictionCone::createData,
          bp::with_custodian_and_ward_postcall<0, 2>(),
          bp::args("self", "data"),
          "Create the contact friction cone residual data.\n\n"
          "Each residual model has its own data that needs to be allocated. "
          "This function\n"
          "returns the allocated data for the contact friction cone residual.\n"
          ":param data: shared data\n"
          ":return residual data.")
      .add_property(
          "id", bp::make_function(&ResidualModelContactFrictionCone::get_id),
          bp::make_function(
              &ResidualModelContactFrictionCone::set_id,
              deprecated<>(
                  "Deprecated. Do not use set_id, instead create a new model")),
          "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelContactFrictionCone::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelContactFrictionCone::set_reference,
          "reference contact friction cone")
      .def(CopyableVisitor<ResidualModelContactFrictionCone>());

  bp::register_ptr_to_python<
      std::shared_ptr<ResidualDataContactFrictionCone> >();

  bp::class_<ResidualDataContactFrictionCone, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactFrictionCone",
      "Data for contact friction cone residual.\n\n",
      bp::init<ResidualModelContactFrictionCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact friction cone residual data.\n\n"
          ":param model: contact friction cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataContactFrictionCone::contact,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactFrictionCone::contact),
          "contact data associated with the current residual")
      .def(CopyableVisitor<ResidualDataContactFrictionCone>());
}

}  // namespace python
}  // namespace crocoddyl

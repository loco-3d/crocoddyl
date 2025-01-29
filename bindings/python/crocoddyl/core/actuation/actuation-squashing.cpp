///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, IRI: CSIC-UPC
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actuation/actuation-squashing.hpp"

#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationSquashing() {
  bp::register_ptr_to_python<std::shared_ptr<ActuationSquashingModel> >();

  bp::class_<ActuationSquashingModel, bp::bases<ActuationModelAbstract> >(
      "ActuationSquashingModel", "Class for squashing an actuation model.\n\n",
      bp::init<std::shared_ptr<ActuationModelAbstract>,
               std::shared_ptr<SquashingModelAbstract>, std::size_t>(
          bp::args("self", "actuation", "squashing", "nu"),
          "Initialize the actuation model with squashing function.\n\n"
          ":param actuation: actuation model to be squashed,\n"
          ":param squashing: squashing function,\n"
          ":param nu: number of controls"))
      .def<void (ActuationSquashingModel::*)(
          const std::shared_ptr<ActuationDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActuationSquashingModel::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the actuation signal from the squashing input u.\n\n"
          "It describes the time-continuos evolution of the actuation model.\n"
          ":param data: actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: squashing function input")
      .def<void (ActuationSquashingModel::*)(
          const std::shared_ptr<ActuationDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActuationSquashingModel::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the actuation model.\n\n"
          "It computes the partial derivatives of the actuation model which "
          "is\n"
          "describes in continouos time. It assumes that calc has been run "
          "first.\n"
          ":param data: actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu).")
      .def("createData", &ActuationSquashingModel::createData, bp::args("self"),
           "Create the actuation squashing data.\n\n"
           "Each actuation model (AM) has its own data that needs to be "
           "allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "squashing",
          bp::make_function(&ActuationSquashingModel::get_squashing,
                            bp::return_value_policy<bp::return_by_value>()),
          "squashing")
      .add_property(
          "actuation",
          bp::make_function(&ActuationSquashingModel::get_actuation,
                            bp::return_value_policy<bp::return_by_value>()),
          "actuation")
      .def(CopyableVisitor<ActuationSquashingModel>());

  bp::register_ptr_to_python<std::shared_ptr<ActuationSquashingData> >();

  bp::class_<ActuationSquashingData, bp::bases<ActuationDataAbstract> >(
      "ActuationSquashingData",
      "Class for actuation datas using squashing functions.\n\n"
      "In crocoddyl, an actuation data contains all the required information "
      "for processing an\n"
      "user-defined actuation model. The actuation data typically is allocated "
      "onces by running\n"
      "model.createData().",
      bp::init<ActuationSquashingModel*>(
          bp::args("self", "model"),
          "Create common data shared between actuation models.\n\n"
          "The actuation data uses the model in order to first process it.\n"
          ":param model: actuation model"))
      .add_property(
          "squashing",
          bp::make_getter(&ActuationSquashingData::squashing,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ActuationSquashingData::squashing),
          "Data of the associated squashing model")
      .add_property(
          "actuation",
          bp::make_getter(&ActuationSquashingData::actuation,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ActuationSquashingData::actuation),
          "Data of the associated actuation model")
      .def(CopyableVisitor<ActuationSquashingData>());
}

}  // namespace python
}  // namespace crocoddyl

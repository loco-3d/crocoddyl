///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/impulse-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualImpulseWrenchCone() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelImpulseWrenchCone> >();

  bp::class_<ResidualModelImpulseWrenchCone, bp::bases<ResidualModelAbstract> >(
      "ResidualModelImpulseWrenchCone", bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, WrenchCone>(
                                            bp::args("self", "state", "id", "fref"),
                                            "Initialize the impulse Wrench cone residual model.\n\n"
                                            ":param state: state of the multibody system\n"
                                            ":param id: reference frame id\n"
                                            ":param fref: impulse wrench cone"))
      .def<void (ResidualModelImpulseWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelImpulseWrenchCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the impulse wrench residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelImpulseWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelImpulseWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelImpulseWrenchCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the impulse wrench cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelImpulseWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelImpulseWrenchCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse wrench cone residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelImpulseWrenchCone::get_id, &ResidualModelImpulseWrenchCone::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelImpulseWrenchCone::get_reference, bp::return_internal_reference<>()),
          &ResidualModelImpulseWrenchCone::set_reference, "reference contact wrench cone");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataImpulseWrenchCone> >();

  bp::class_<ResidualDataImpulseWrenchCone, bp::bases<ResidualDataAbstract> >(
      "ResidualDataImpulseWrenchCone", "Data for impulse Wrench cone residual.\n\n",
      bp::init<ResidualModelImpulseWrenchCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create impulse Wrench cone residual data.\n\n"
          ":param model: impulse Wrench cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulse",
          bp::make_getter(&ResidualDataImpulseWrenchCone::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataImpulseWrenchCone::impulse),
          "impulse data associated with the current residual");
}

}  // namespace python
}  // namespace crocoddyl

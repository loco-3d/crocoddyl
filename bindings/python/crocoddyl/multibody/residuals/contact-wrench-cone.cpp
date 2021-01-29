///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactWrenchCone() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelContactWrenchCone> >();

  bp::class_<ResidualModelContactWrenchCone, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactWrenchCone",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, WrenchCone, std::size_t>(
          bp::args("self", "state", "id", "fref", "nu"),
          "Initialize the contact wrench cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: contact wrench cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, WrenchCone>(
          bp::args("self", "state", "id", "fref"),
          "Initialize the contact wrench cone residual model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: contact wrench cone"))
      .def<void (ResidualModelContactWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactWrenchCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact wrench cone residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelContactWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactWrenchCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact wrench cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelContactWrenchCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactWrenchCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact wrench cone residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelContactWrenchCone::get_id, &ResidualModelContactWrenchCone::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelContactWrenchCone::get_reference, bp::return_internal_reference<>()),
          &ResidualModelContactWrenchCone::set_reference, "reference contact wrench cone");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataContactWrenchCone> >();

  bp::class_<ResidualDataContactWrenchCone, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactWrenchCone", "Data for contact wrench cone residual.\n\n",
      bp::init<ResidualModelContactWrenchCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact wrench cone residual data.\n\n"
          ":param model: contact wrench cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataContactWrenchCone::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactWrenchCone::contact),
          "contact data associated with the current residual");
}

}  // namespace python
}  // namespace crocoddyl

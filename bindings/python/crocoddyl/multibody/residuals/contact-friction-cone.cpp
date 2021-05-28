///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactFrictionCone() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelContactFrictionCone> >();

  bp::class_<ResidualModelContactFrictionCone, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactFrictionCone",
      "This residual function is defined as r = A*f, where A, f describe the linearized friction cone and\n"
      "the spatial force, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, FrictionCone, std::size_t>(
          bp::args("self", "state", "id", "fref", "nu"),
          "Initialize the contact friction cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: frame friction cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, FrictionCone>(
          bp::args("self", "state", "id", "fref"),
          "Initialize the contact friction cone residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: frame friction cone"))
      .def<void (ResidualModelContactFrictionCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactFrictionCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact friction cone residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelContactFrictionCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactFrictionCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactFrictionCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact friction cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelContactFrictionCone::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactFrictionCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact friction cone residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the contact friction cone residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelContactFrictionCone::get_id, &ResidualModelContactFrictionCone::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelContactFrictionCone::get_reference, bp::return_internal_reference<>()),
          &ResidualModelContactFrictionCone::set_reference, "reference contact friction cone");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataContactFrictionCone> >();

  bp::class_<ResidualDataContactFrictionCone, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactFrictionCone", "Data for contact friction cone residual.\n\n",
      bp::init<ResidualModelContactFrictionCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact friction cone residual data.\n\n"
          ":param model: contact friction cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataContactFrictionCone::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactFrictionCone::contact),
          "contact data associated with the current residual");
}

}  // namespace python
}  // namespace crocoddyl

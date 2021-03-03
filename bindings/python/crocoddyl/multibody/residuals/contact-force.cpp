///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactForce() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelContactForce> >();

  bp::class_<ResidualModelContactForce, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactForce",
      "This residual function is defined as r = f-fref, where f,fref describe the current and reference\n"
      "the spatial forces, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::Force, std::size_t, std::size_t>(
          bp::args("self", "state", "id", "fref", "nc", "nu"),
          "Initialize the contact force residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nc: dimension of the contact force (nc <= 6)\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::Force, std::size_t>(
          bp::args("self", "state", "id", "fref", "nc"),
          "Initialize the contact force residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param nc: dimension of the contact force (nc <= 6)\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"))
      .def<void (ResidualModelContactForce::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactForce::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact force residual.\n\n"
          ":param data: residual data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (ResidualModelContactForce::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactForce::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactForce::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact force residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (ResidualModelContactForce::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                               const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactForce::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact force residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the contact force residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelContactForce::get_id, &ResidualModelContactForce::set_id, "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelContactForce::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelContactForce::set_reference, "reference spatial force");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataContactForce> >();

  bp::class_<ResidualDataContactForce, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactForce", "Data for contact force residual.\n\n",
      bp::init<ResidualModelContactForce*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact force residual data.\n\n"
          ":param model: contact force residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataContactForce::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactForce::contact), "contact data associated with the current residual");
}

}  // namespace python
}  // namespace crocoddyl

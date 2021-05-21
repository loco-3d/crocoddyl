///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactControlGrav() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelContactControlGrav> >();

  bp::class_<ResidualModelContactControlGrav, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactControlGrav",
      "This residual function defines a residual vector as r = u - g(q,fext),\n"
      "with u as the control, q as the position, fext as the external forces and g as the gravity vector in contact",
      bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the contact control-gravity residual model.\n\n"
          ":param state: state description\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                        "Initialize the contact control-gravity residual model.\n\n"
                                                        "The default nu is obtained from state.nv.\n"
                                                        ":param state: state description"))
      .def<void (ResidualModelContactControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelContactControlGrav::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact control-gravity residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelContactControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelContactControlGrav::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobias of the contact control-gravity residual.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelContactControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                                     const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactControlGrav::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact control-gravity residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataContactControlGrav> >();

  bp::class_<ResidualDataContactControlGrav, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactControlGrav", "Data for control gravity residual in contact.\n\n",
      bp::init<ResidualModelContactControlGrav *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create contact control-gravity gravity contact residual data.\n\n"
          ":param model: control gravity residual model in contact\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio", bp::make_getter(&ResidualDataContactControlGrav::pinocchio),
                    "Pinocchio data used for internal computations")
      .add_property("actuation",
                    bp::make_getter(&ResidualDataContactControlGrav::actuation, bp::return_internal_reference<>()),
                    "actuation model")
      .add_property("fext", bp::make_getter(&ResidualDataContactControlGrav::fext, bp::return_internal_reference<>()),
                    "external spatial forces");
}

}  // namespace python
}  // namespace crocoddyl

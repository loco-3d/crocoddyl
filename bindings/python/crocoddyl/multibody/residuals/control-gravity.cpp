///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/control-gravity.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualControlGrav() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelControlGrav> >();

  bp::class_<ResidualModelControlGrav, bp::bases<ResidualModelAbstract> >(
      "ResidualModelControlGrav",
      "This residual function is defined as r = a(u) - g(q), where a(u)\n"
      "is the actuated torque; and q, g(q) are the generalized position\n"
      "and gravity vector, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(bp::args("self", "state", "nu"),
                                                               "Initialize the control-gravity residual model.\n\n"
                                                               ":param state: state description\n"
                                                               ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                        "Initialize the control-gravity residual model.\n\n"
                                                        "The default nu is obtained from state.nv.\n"
                                                        ":param state: state description"))
      .def<void (ResidualModelControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelControlGrav::calc, bp::args("self", "data", "x", "u"),
          "Compute the control residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelControlGrav::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control residual.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelControlGrav::*)(const boost::shared_ptr<ResidualDataAbstract> &,
                                              const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelControlGrav::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataControlGrav> >();

  bp::class_<ResidualDataControlGrav, bp::bases<ResidualDataAbstract> >(
      "ResidualDataControlGrav", "Data for control gravity residual.\n\n",
      bp::init<ResidualModelControlGrav *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create control gravity residual data.\n\n"
          ":param model: control gravity residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio", bp::make_getter(&ResidualDataControlGrav::pinocchio),
                    "Pinocchio data used for internal computations")
      .add_property("actuation",
                    bp::make_getter(&ResidualDataControlGrav::actuation, bp::return_internal_reference<>()),
                    "actuation model");
}

}  // namespace python
}  // namespace crocoddyl

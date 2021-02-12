///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/residuals/control.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualControl() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelControl> >();

  bp::class_<ResidualModelControl, bp::bases<ResidualModelAbstract> >(
      "ResidualModelControl",
      "This residual function defines a residual vector as r = u - uref, with u and uref as the current and reference "
      "control, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(bp::args("self", "state", "uref"),
                                                                  "Initialize the control residual model.\n\n"
                                                                  ":param state: state description\n"
                                                                  ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the control residual model.\n\n"
          "The default reference control is obtained from np.zero(nu).\n"
          ":param state: state description\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(bp::args("self", "state"),
                                                       "Initialize the control residual model.\n\n"
                                                       "The default reference control is obtained from np.zero(nu).\n"
                                                       ":param state: state description"))
      .def<void (ResidualModelControl::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelControl::calc, bp::args("self", "data", "x", "u"),
          "Compute the control residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelControl::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelControl::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelControl::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelControl::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelControl::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("reference",
                    bp::make_function(&ResidualModelControl::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelControl::set_reference, "reference control vector");
}

}  // namespace python
}  // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualCoMPosition() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelCoMPosition> >();

  bp::class_<ResidualModelCoMPosition, bp::bases<ResidualModelAbstract> >(
      "ResidualModelCoMPosition",
      "This residual function defines the CoM tracking as r = c - cref, with c and cref as the current and reference "
      "CoM position, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d, std::size_t>(
          bp::args("self", "state", "cref", "nu"),
          "Initialize the CoM position residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d>(
          bp::args("self", "state", "cref"),
          "Initialize the CoM position residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position"))
      .def<void (ResidualModelCoMPosition::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelCoMPosition::calc, bp::args("self", "data", "x", "u"),
          "Compute the CoM position residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelCoMPosition::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelCoMPosition::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelCoMPosition::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the CoM position residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelCoMPosition::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelCoMPosition::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the CoM position residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("reference",
                    bp::make_function(&ResidualModelCoMPosition::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelCoMPosition::set_reference, "reference CoM position");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataCoMPosition> >();

  bp::class_<ResidualDataCoMPosition, bp::bases<ResidualDataAbstract> >(
      "ResidualDataCoMPosition", "Data for CoM position residual.\n\n",
      bp::init<ResidualModelCoMPosition*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create CoM position residual data.\n\n"
          ":param model: CoM position residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataCoMPosition::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data");
}

}  // namespace python
}  // namespace crocoddyl

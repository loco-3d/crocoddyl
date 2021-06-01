///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameVelocity() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameVelocity> >();

  bp::class_<ResidualModelFrameVelocity, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFrameVelocity",
      "This residual function defines r = v - vref, with v and vref as the current and reference\n"
      "frame velocities, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, std::size_t, pinocchio::Motion, pinocchio::ReferenceFrame,
               std::size_t>(bp::args("self", "state", "id", "velocity", "type", "nu"),
                            "Initialize the frame velocity residual model.\n\n"
                            ":param state: state of the multibody system\n"
                            ":param residual: residual model\n"
                            ":param id: reference frame id\n"
                            ":param velocity: reference velocity\n"
                            ":param type: reference type of velocity\n"
                            ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::Motion,
                    pinocchio::ReferenceFrame>(bp::args("self", "state", "id", "velocity", "type"),
                                               "Initialize the frame velocity residual model.\n\n"
                                               ":param state: state of the multibody system\n"
                                               ":param residual: residual model\n"
                                               ":param id: reference frame id\n"
                                               ":param velocity: reference velocity\n"
                                               ":param type: reference type of velocity"))
      .def<void (ResidualModelFrameVelocity::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameVelocity::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame velocity residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelFrameVelocity::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameVelocity::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameVelocity::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame velocity residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelFrameVelocity::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameVelocity::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame velocity residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the frame velocity residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameVelocity::get_id, &ResidualModelFrameVelocity::set_id,
                    "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelFrameVelocity::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelFrameVelocity::set_reference, "reference velocity")
      .add_property("type", &ResidualModelFrameVelocity::get_type, &ResidualModelFrameVelocity::set_type,
                    "reference type of velocity");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameVelocity> >();

  bp::class_<ResidualDataFrameVelocity, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFrameVelocity", "Data for frame velocity residual.\n\n",
      bp::init<ResidualModelFrameVelocity*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame velocity residual data.\n\n"
          ":param model: frame Velocity residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameVelocity::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data");
}

}  // namespace python
}  // namespace crocoddyl

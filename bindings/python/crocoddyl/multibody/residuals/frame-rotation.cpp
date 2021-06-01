///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameRotation() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameRotation> >();

  bp::class_<ResidualModelFrameRotation, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFrameRotation",
      "This residual function is defined as r = R - Rref, with R and Rref as the current and reference\n"
      "frame rotations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, Eigen::Matrix3d, std::size_t>(
          bp::args("self", "state", "id", "Rref", "nu"),
          "Initialize the frame rotation residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, Eigen::Matrix3d>(
          bp::args("self", "state", "id", "Rref"),
          "Initialize the frame rotation residual model.\n\n"
          "The default nu value is obtained from model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param Rref: reference frame rotation"))
      .def<void (ResidualModelFrameRotation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameRotation::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame rotation residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelFrameRotation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameRotation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameRotation::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame rotation residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelFrameRotation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameRotation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame rotation residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the frame rotation residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameRotation::get_id, &ResidualModelFrameRotation::set_id,
                    "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelFrameRotation::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelFrameRotation::set_reference, "reference frame rotation");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameRotation> >();

  bp::class_<ResidualDataFrameRotation, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFrameRotation", "Data for frame rotation residual.\n\n",
      bp::init<ResidualModelFrameRotation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame rotation residual data.\n\n"
          ":param model: frame rotation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameRotation::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("r", bp::make_getter(&ResidualDataFrameRotation::r, bp::return_internal_reference<>()),
                    "residual residual")
      .add_property("rRf", bp::make_getter(&ResidualDataFrameRotation::rRf, bp::return_internal_reference<>()),
                    "rotation error of the frame")
      .add_property("rJf", bp::make_getter(&ResidualDataFrameRotation::rJf, bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf", bp::make_getter(&ResidualDataFrameRotation::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

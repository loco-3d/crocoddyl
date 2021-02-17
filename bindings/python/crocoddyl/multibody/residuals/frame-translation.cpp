///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameTranslation() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameTranslation> >();

  bp::class_<ResidualModelFrameTranslation, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFrameTranslation",
      "This residual function defines the the frame translation tracking as as r = t - tref, with t and tref as the\n"
      "current and reference frame translations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, Eigen::Vector3d, std::size_t>(
          bp::args("self", "state", "id", "xref", "nu"),
          "Initialize the frame translation residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, Eigen::Vector3d>(
          bp::args("self", "state", "id", "xref"),
          "Initialize the frame translation residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param xref: reference frame translation"))
      .def<void (ResidualModelFrameTranslation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameTranslation::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame translation residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelFrameTranslation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameTranslation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameTranslation::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame translation residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelFrameTranslation::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameTranslation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame translation residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the frame translation residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameTranslation::get_id, &ResidualModelFrameTranslation::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelFrameTranslation::get_reference, bp::return_internal_reference<>()),
          &ResidualModelFrameTranslation::set_reference, "reference frame translation");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameTranslation> >();

  bp::class_<ResidualDataFrameTranslation, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFrameTranslation", "Data for frame translation residual.\n\n",
      bp::init<ResidualModelFrameTranslation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame translation residual data.\n\n"
          ":param model: frame translation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameTranslation::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("fJf", bp::make_getter(&ResidualDataFrameTranslation::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

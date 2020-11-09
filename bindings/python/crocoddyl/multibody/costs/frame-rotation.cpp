///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameRotation() {
  bp::class_<CostModelFrameRotation, bp::bases<CostModelAbstract> >(
      "CostModelFrameRotation",
      "This cost function defines a residual vector as r = R - Rref, with R and Rref as the current and reference "
      "frame rotations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameRotation, int>(
          bp::args("self", "state", "activation", "Rref", "nu"),
          "Initialize the frame rotation cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameRotation>(
          bp::args("self", "state", "activation", "Rref"),
          "Initialize the frame rotation cost model.\n\n"
          "The default nu value is obtained from model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Rref: reference frame rotation"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation, int>(
          bp::args("self", "state", "Rref", "nu"),
          "Initialize the frame rotation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation>(
          bp::args("self", "state", "Rref"),
          "Initialize the frame rotation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation"))
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelFrameRotation::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame rotation cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelFrameRotation::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame rotation cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelFrameRotation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame rotation cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelFrameRotation::get_reference<FrameRotation>,
                    &CostModelFrameRotation::set_reference<FrameRotation>, "reference rotation and index")
      .add_property("reference_rotation", &CostModelFrameRotation::get_reference<Eigen::Matrix3d>,
                    &CostModelFrameRotation::set_reference<Eigen::Matrix3d>, "reference rotation")
      .add_property("reference_id", &CostModelFrameRotation::get_reference<FrameIndex>,
                    &CostModelFrameRotation::set_reference<FrameIndex>, "reference index")
      .add_property("Rref",
                    bp::make_function(&CostModelFrameRotation::get_reference<FrameRotation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFrameRotation::set_reference<FrameRotation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame rotation");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataFrameRotation> >();

  bp::class_<CostDataFrameRotation, bp::bases<CostDataAbstract> >(
      "CostDataFrameRotation", "Data for frame rotation cost.\n\n",
      bp::init<CostModelFrameRotation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame rotation cost data.\n\n"
          ":param model: frame rotation cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("r", bp::make_getter(&CostDataFrameRotation::r, bp::return_internal_reference<>()),
                    "cost residual")
      .add_property("rRf", bp::make_getter(&CostDataFrameRotation::rRf, bp::return_internal_reference<>()),
                    "rotation error of the frame")
      .add_property("J", bp::make_getter(&CostDataFrameRotation::J, bp::return_internal_reference<>()),
                    "Jacobian at the error point")
      .add_property("rJf", bp::make_getter(&CostDataFrameRotation::rJf, bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf", bp::make_getter(&CostDataFrameRotation::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

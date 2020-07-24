///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameVelocity() {
  bp::class_<CostModelFrameVelocity, bp::bases<CostModelAbstract> >(
      "CostModelFrameVelocity",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameMotion, int>(
          bp::args("self", "state", "activation", "vref", "nu"),
          "Initialize the frame velocity cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param vref: reference frame velocity\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameMotion>(
          bp::args("self", "state", "activation", "vref"),
          "Initialize the frame velocity cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param vref: reference frame velocity"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameMotion, int>(
          bp::args("self", "state", "vref", "nu"),
          "Initialize the frame velocity cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param vref: reference frame velocity\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameMotion>(
          bp::args("self", "state", "vref"),
          "Initialize the frame velocity cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param vref: reference frame velocity"))
      .def<void (CostModelFrameVelocity::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelFrameVelocity::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame velocity cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelFrameVelocity::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelFrameVelocity::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelFrameVelocity::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame velocity cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelFrameVelocity::*)(const boost::shared_ptr<CostDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelFrameVelocity::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame velocity cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", bp::make_function(&CostModelFrameVelocity::get_reference<FrameMotion>),
                    &CostModelFrameVelocity::set_reference<FrameMotion>, "reference frame velocity")
      .add_property("vref", bp::make_function(&CostModelFrameVelocity::get_reference<FrameMotion>),
                    &CostModelFrameVelocity::set_reference<FrameMotion>, "reference frame velocity");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataFrameVelocity> >();

  bp::class_<CostDataFrameVelocity, bp::bases<CostDataAbstract> >(
      "CostDataFrameVelocity", "Data for frame velocity cost.\n\n",
      bp::init<CostModelFrameVelocity*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame velocity cost data.\n\n"
          ":param model: frame Velocity cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("joint", bp::make_getter(&CostDataFrameVelocity::joint), "joint index")
      .add_property("vr", bp::make_getter(&CostDataFrameVelocity::vr, bp::return_value_policy<bp::return_by_value>()),
                    "error velocity of the frame")
      .add_property("fXj",
                    bp::make_getter(&CostDataFrameVelocity::fXj, bp::return_value_policy<bp::return_by_value>()),
                    "action matrix from contact to local frames")
      .add_property("dv_dq", bp::make_getter(&CostDataFrameVelocity::dv_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("dv_dv", bp::make_getter(&CostDataFrameVelocity::dv_dv, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity");
}

}  // namespace python
}  // namespace crocoddyl

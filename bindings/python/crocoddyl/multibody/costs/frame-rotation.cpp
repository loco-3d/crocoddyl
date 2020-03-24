///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameRotation() {
  bp::class_<CostModelFrameRotation, bp::bases<CostModelAbstract> >(
      "CostModelFrameRotation",
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
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Rref: reference frame rotation"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation, int>(
          bp::args("self", "state", "Rref", "nu"),
          "Initialize the frame rotation cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation>(
          bp::args("self", "state", "Rref"),
          "Initialize the frame rotation cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation"))
      .def("calc", &CostModelFrameRotation::calc_wrap,
           CostModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                "Compute the frame rotation cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                            const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameRotation::calcDiff_wrap, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame rotation cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelFrameRotation::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameRotation::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &CostModelFrameRotation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame rotation cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    bp::make_function(&CostModelFrameRotation::get_Rref, bp::return_internal_reference<>()),
                    &CostModelFrameRotation::set_reference<FrameRotation>, "reference frame rotation")
      .add_property("Rref", bp::make_function(&CostModelFrameRotation::get_Rref, bp::return_internal_reference<>()),
                    &CostModelFrameRotation::set_Rref, "reference frame rotation");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataFrameRotation> >();

  bp::class_<CostDataFrameRotation, bp::bases<CostDataAbstract> >(
      "CostDataFrameRotation", "Data for frame rotation cost.\n\n",
      bp::init<CostModelFrameRotation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame rotation cost data.\n\n"
          ":param model: frame rotation cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("r", bp::make_getter(&CostDataFrameRotation::r, bp::return_value_policy<bp::return_by_value>()),
                    "cost residual")
      .add_property("rRf",
                    bp::make_getter(&CostDataFrameRotation::rRf, bp::return_value_policy<bp::return_by_value>()),
                    "rotation error of the frame")
      .add_property("J", bp::make_getter(&CostDataFrameRotation::J, bp::return_value_policy<bp::return_by_value>()),
                    "Jacobian at the error point")
      .add_property("rJf",
                    bp::make_getter(&CostDataFrameRotation::rJf, bp::return_value_policy<bp::return_by_value>()),
                    "error Jacobian of the frame")
      .add_property("fJf",
                    bp::make_getter(&CostDataFrameRotation::fJf, bp::return_value_policy<bp::return_by_value>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

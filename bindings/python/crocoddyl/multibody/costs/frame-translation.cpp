///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameTranslation() {
  bp::class_<CostModelFrameTranslation, bp::bases<CostModelAbstract> >(
      "CostModelFrameTranslation",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameTranslation, int>(
          bp::args("self", "state", "activation", "xref", "nu"),
          "Initialize the frame translation cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameTranslation>(
          bp::args("self", "state", "activation", "xref"),
          "Initialize the frame translation cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference frame translation"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation, int>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the frame translation cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3).\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation>(
          bp::args("self", "state", "xref"),
          "Initialize the frame translation cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation"))
      .def("calc", &CostModelFrameTranslation::calc_wrap,
           CostModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                "Compute the frame translation cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelFrameTranslation::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                               const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameTranslation::calcDiff_wrap, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame translation cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelFrameTranslation::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameTranslation::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &CostModelFrameTranslation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame translation cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    bp::make_function(&CostModelFrameTranslation::get_xref, bp::return_internal_reference<>()),
                    &CostModelFrameTranslation::set_reference<FrameTranslation>, "reference frame translation")
      .add_property("xref", bp::make_function(&CostModelFrameTranslation::get_xref, bp::return_internal_reference<>()),
                    &CostModelFrameTranslation::set_xref, "reference frame translation");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataFrameTranslation> >();

  bp::class_<CostDataFrameTranslation, bp::bases<CostDataAbstract> >(
      "CostDataFrameTranslation", "Data for frame translation cost.\n\n",
      bp::init<CostModelFrameTranslation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame translation cost data.\n\n"
          ":param model: frame translation cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("J", bp::make_getter(&CostDataFrameTranslation::J, bp::return_value_policy<bp::return_by_value>()),
                    "Jacobian at the error point")
      .add_property("fJf",
                    bp::make_getter(&CostDataFrameTranslation::fJf, bp::return_value_policy<bp::return_by_value>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

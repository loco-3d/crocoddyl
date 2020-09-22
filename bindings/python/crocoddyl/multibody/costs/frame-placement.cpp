///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFramePlacement() {
  bp::class_<CostModelFramePlacement, bp::bases<CostModelAbstract> >(
      "CostModelFramePlacement",
      "This cost function defines a residual vector as r = p - pref, with p and pref as the current and reference "
      "frame placements, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FramePlacement, int>(
          bp::args("self", "state", "activation", "Mref", "nu"),
          "Initialize the frame placement cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FramePlacement>(
          bp::args("self", "state", "activation", "Mref"),
          "Initialize the frame placement cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Mref: reference frame placement"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FramePlacement, int>(
          bp::args("self", "state", "Mref", "nu"),
          "Initialize the frame placement cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FramePlacement>(
          bp::args("self", "state", "Mref"),
          "Initialize the frame placement cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement"))
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelFramePlacement::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame placement cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame placement cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelFramePlacement::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame placement cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelFramePlacement::get_reference<FramePlacement>,
                    &CostModelFramePlacement::set_reference<FramePlacement>, "reference frame placement")
      .add_property("Mref",
                    bp::make_function(&CostModelFramePlacement::get_reference<FramePlacement>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFramePlacement::set_reference<FramePlacement>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame placement");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataFramePlacement> >();

  bp::class_<CostDataFramePlacement, bp::bases<CostDataAbstract> >(
      "CostDataFramePlacement", "Data for frame placement cost.\n\n",
      bp::init<CostModelFramePlacement*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame placement cost data.\n\n"
          ":param model: frame placement cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("r", bp::make_getter(&CostDataFramePlacement::r, bp::return_internal_reference<>()),
                    "cost residual")
      .add_property("rMf",
                    bp::make_getter(&CostDataFramePlacement::rMf, bp::return_value_policy<bp::return_by_value>()),
                    "error frame placement of the frame")
      .add_property("J", bp::make_getter(&CostDataFramePlacement::J, bp::return_internal_reference<>()),
                    "Jacobian at the error point")
      .add_property("rJf", bp::make_getter(&CostDataFramePlacement::rJf, bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf", bp::make_getter(&CostDataFramePlacement::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl

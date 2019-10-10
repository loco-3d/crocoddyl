///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostState() {
  bp::class_<CostModelState, bp::bases<CostModelAbstract> >(
      "CostModelState",
      bp::init<boost::shared_ptr<StateMultibody>, ActivationModelAbstract&, Eigen::VectorXd, int>(
          bp::args(" self", " state", " activation=crocoddyl.ActivationModelQuad(state.ndx)", " xref=state.zero()",
                   " nu=model.nv"),
          "Initialize the state cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference state\n"
          ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::VectorXd, int>(
          bp::args(" self", " state", " xref", " nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx).\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference state\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, ActivationModelAbstract&, Eigen::VectorXd>(
          bp::args(" self", " state", " activation", " xref"),
          "Initialize the state cost model.\n\n"
          "For this case the default nu values is model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::VectorXd>(
          bp::args(" self", " state", " xref"),
          "Initialize the state cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx),\n"
          "and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, ActivationModelAbstract&, int>(
          bp::args(" self", " state", " activation", " nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero().\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args(" self", " state", " nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), and the default activation\n"
          "model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx)\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, ActivationModelAbstract&>(
          bp::args(" self", " state", " activation"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args(" self", " state"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), the default activation\n"
          "model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system"))
      .def("calc", &CostModelState::calc_wrap,
           CostModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                "Compute the state cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelState::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the state cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>("calcDiff", &CostModelState::calcDiff_wrap,
                                                             bp::args(" self", " data", " x", " u"))
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelState::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelState::calcDiff_wrap, bp::args(" self", " data", " x", " recalc"))
      .add_property("xref",
                    bp::make_function(&CostModelState::get_xref, bp::return_value_policy<bp::return_by_value>()),
                    "reference state")
      .def("createData", &CostModelState::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the state cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return cost data.");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

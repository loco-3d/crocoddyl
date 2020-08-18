///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostState() {
  bp::class_<CostModelState, bp::bases<CostModelAbstract> >(
      "CostModelState",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd, int>(
          bp::args("self", "state", "activation", "xref", " nu=model.nv"),
          "Initialize the state cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::VectorXd, int>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx).\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference state\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "activation", "xref"),
          "Initialize the state cost model.\n\n"
          "For this case the default nu values is model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx),\n"
          "and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero().\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), and the default activation\n"
          "model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx)\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "For this case the default xref is the zeros state, i.e. state.zero(), the default activation\n"
          "model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system"))
      .def<void (CostModelState::*)(
          const boost::shared_ptr<CostDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelState::calc, bp::args("self", "data", "x", "u"),
                                                     "Compute the state cost.\n\n"
                                                     ":param data: cost data\n"
                                                     ":param x: time-discrete state vector\n"
                                                     ":param u: time-discrete control input")
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                               bp::args("self", "data", "x"))
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelState::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelState::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the state cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelState::get_reference<Eigen::VectorXd>,
                    &CostModelState::set_reference<Eigen::VectorXd>, "reference state")
      .add_property("xref",
                    bp::make_function(&CostModelState::get_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelState::set_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference state");
}

}  // namespace python
}  // namespace crocoddyl

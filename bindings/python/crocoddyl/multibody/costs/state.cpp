///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostState() {  // TODO: Remove once the deprecated update call has been removed in a future
                          // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelState> >();

  bp::class_<CostModelState, bp::bases<CostModelResidual> >(
      "CostModelState",
      "This cost function defines a residual vector as r = x - xref, with x and xref as the current and reference "
      "state, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd, int>(
          bp::args("self", "state", "activation", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd, int>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state description\n"
          ":param xref: reference state\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "activation", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, int>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference state "
          "is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference state "
          "is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description"))
      .def<void (CostModelState::*)(const boost::shared_ptr<CostDataAbstract>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelState::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .add_property("reference", &CostModelState::get_reference<Eigen::VectorXd>,
                    &CostModelState::set_reference<Eigen::VectorXd>, "reference state")
      .add_property("xref",
                    bp::make_function(&CostModelState::get_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelState::set_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference state");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

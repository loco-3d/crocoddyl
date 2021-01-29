///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/costs/control.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostControl() {
  bp::class_<CostModelControl, bp::bases<CostModelAbstract>>(
      "CostModelControl",
      "This cost function defines a residual vector as r = u - uref, with u "
      "and uref as the current and reference "
      "control, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>,
               boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "activation", "uref"),
          "Initialize the control cost model.\n\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateAbstract>,
                    boost::shared_ptr<ActivationModelAbstract>>(
          bp::args("self", "state", "activation"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu), with nu "
          "obtained from activation.nr.\n"
          ":param state: state description\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateAbstract>,
                    boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu).\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "uref"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2).\n"
          ":param state: state description\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateAbstract>>(
          bp::args("self", "state"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(state.nv).\n"
          ":param state: state description"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, int>(
          bp::args("self", "state", "nu"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(nu)\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def<void (CostModelControl::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelControl::calc, bp::args("self", "data", "x", "u"),
          "Compute the control cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelControl::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelControl::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelControl::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelControl::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &CostModelControl::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    &CostModelControl::get_reference<Eigen::VectorXd>,
                    &CostModelControl::set_reference<Eigen::VectorXd>,
                    "reference control vector")
      .add_property(
          "uref",
          bp::make_function(&CostModelControl::get_reference<Eigen::VectorXd>,
                            deprecated<>("Deprecated. Use reference.")),
          bp::make_function(&CostModelControl::set_reference<Eigen::VectorXd>,
                            deprecated<>("Deprecated. Use reference.")),
          "reference control vector");
}

} // namespace python
} // namespace crocoddyl

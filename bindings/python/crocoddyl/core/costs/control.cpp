///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/costs/control.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostControl() {  // TODO: Remove once the deprecated update call has been removed in a future
                            // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelControl> >();

  bp::class_<CostModelControl, bp::bases<CostModelResidual> >(
      "CostModelControl",
      "This cost function defines a residual vector as r = u - uref, with u and uref as the current and reference "
      "control, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "activation", "uref"),
          "Initialize the control cost model.\n\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu), with nu obtained from activation.nr.\n"
          ":param state: state description\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu).\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "uref"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state description\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(state.nv).\n"
          ":param state: state description"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, int>(
          bp::args("self", "state", "nu"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(nu)\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def<void (CostModelControl::*)(const boost::shared_ptr<CostDataAbstract>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelControl::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .add_property("reference", &CostModelControl::get_reference<Eigen::VectorXd>,
                    &CostModelControl::set_reference<Eigen::VectorXd>, "reference control vector")
      .add_property("uref",
                    bp::make_function(&CostModelControl::get_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelControl::set_reference<Eigen::VectorXd>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference control vector");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

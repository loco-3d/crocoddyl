///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/control-gravity.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostControlGrav() {
  bp::class_<CostModelControlGrav, bp::bases<CostModelAbstract>>(
      "CostModelControlGrav",
      "This cost function defines a residual vector as r = a(u) - g(q), "
      "where q, g(q) are the generalized position and gravity vector, respectively,"
      "and a(u) is the actuated torque.",
      bp::init<boost::shared_ptr<StateMultibody>,
               boost::shared_ptr<ActivationModelAbstract>,
               boost::shared_ptr<ActuationModelFull>>(
          bp::args("self", "state", "activation", "actuation"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu), with nu "
          "obtained from activation.nr.\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param actuation: actuation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody>,
                    boost::shared_ptr<ActuationModelFull>>(
          bp::args("self", "state", "actuation"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(state.nv).\n"
          ":param state: state description\n"
          ":param actuation: actuation model"))
      .def<void (CostModelControlGrav::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelControlGrav::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the control cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelControlGrav::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelControlGrav::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelControlGrav::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelControlGrav::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &CostModelControlGrav::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");
  bp::class_<CostDataControlGrav, bp::bases<CostDataAbstract>>(
      "CostDataControlGrav", "Data for control gravity cost.\n\n",
      bp::init<CostModelControlGrav *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create control gravity cost data.\n\n"
          ":param model: control gravity cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("dg_dq",
       bp::make_getter(&CostDataControlGrav::dg_dq,
       bp::return_internal_reference<>()),
       "Partial derivative of gravity vector with respect to q");
}

} // namespace python
} // namespace crocoddyl

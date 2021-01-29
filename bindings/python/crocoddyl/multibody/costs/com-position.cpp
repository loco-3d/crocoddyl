///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/com-position.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostCoMPosition() {
  bp::class_<CostModelCoMPosition, bp::bases<CostModelAbstract>>(
      "CostModelCoMPosition",
      "This cost function defines a residual vector as r = c - cref, with c "
      "and cref as the current and reference "
      "CoM position, respetively.",
      bp::init<boost::shared_ptr<StateMultibody>,
               boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d,
               int>(bp::args("self", "state", "activation", "cref", "nu"),
                    "Initialize the CoM position cost model.\n\n"
                    ":param state: state of the multibody system\n"
                    ":param activation: activation model\n"
                    ":param cref: reference CoM position\n"
                    ":param nu: dimension of control vector"))
      .def(
          bp::init<boost::shared_ptr<StateMultibody>,
                   boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d>(
              bp::args("self", "state", "activation", "cref"),
              "Initialize the CoM position cost model.\n\n"
              "The default nu is obtained from state.nv.\n"
              ":param state: state of the multibody system\n"
              ":param activation: activation model\n"
              ":param cref: reference CoM position"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d, int>(
          bp::args("self", "state", "cref", "nu"),
          "Initialize the CoM position cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d>(
          bp::args("self", "state", "cref"),
          "Initialize the CoM position cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. "
          "a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position"))
      .def<void (CostModelCoMPosition::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelCoMPosition::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the CoM position cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelCoMPosition::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelCoMPosition::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelCoMPosition::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the CoM position cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelCoMPosition::*)(
          const boost::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &CostModelCoMPosition::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the CoM position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    &CostModelCoMPosition::get_reference<Eigen::Vector3d>,
                    &CostModelCoMPosition::set_reference<Eigen::Vector3d>,
                    "reference CoM position")
      .add_property("cref",
                    bp::make_function(
                        &CostModelCoMPosition::get_reference<Eigen::Vector3d>,
                        deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(
                        &CostModelCoMPosition::set_reference<Eigen::Vector3d>,
                        deprecated<>("Deprecated. Use reference.")),
                    "reference CoM position");
}

} // namespace python
} // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualState() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelState> >();

  bp::class_<ResidualModelState, bp::bases<ResidualModelAbstract> >(
      "ResidualModelState",
      "This cost function defines a residual vector as r = x - xref, with x and xref as the current and reference "
      "state, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelState::calc, bp::args("self", "data", "x", "u"),
          "Compute the state cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelState::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .add_property("reference",
                    bp::make_function(&ResidualModelState::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelState::set_reference, "reference state");
}

}  // namespace python
}  // namespace crocoddyl

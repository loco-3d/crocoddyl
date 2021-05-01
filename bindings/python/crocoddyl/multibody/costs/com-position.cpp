///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostCoMPosition() {  // TODO: Remove once the deprecated update call has been removed in a future
                                // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelCoMPosition> >();

  bp::class_<CostModelCoMPosition, bp::bases<CostModelResidual> >(
      "CostModelCoMPosition",
      "This cost function defines a residual vector as r = c - cref, with c and cref as the current and reference "
      "CoM position, respetively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d, int>(
          bp::args("self", "state", "activation", "cref", "nu"),
          "Initialize the CoM position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d>(
          bp::args("self", "state", "activation", "cref"),
          "Initialize the CoM position cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param cref: reference CoM position"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d, int>(
          bp::args("self", "state", "cref", "nu"),
          "Initialize the CoM position cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d>(
          bp::args("self", "state", "cref"),
          "Initialize the CoM position cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position"))
      .add_property("reference", &CostModelCoMPosition::get_reference<Eigen::Vector3d>,
                    &CostModelCoMPosition::set_reference<Eigen::Vector3d>, "reference CoM position")
      .add_property("cref",
                    bp::make_function(&CostModelCoMPosition::get_reference<Eigen::Vector3d>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelCoMPosition::set_reference<Eigen::Vector3d>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference CoM position");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameVelocity() {  // TODO: Remove once the deprecated update call has been removed in a future
                                  // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelFrameVelocity> >();

  bp::class_<CostModelFrameVelocity, bp::bases<CostModelResidual> >(
      "CostModelFrameVelocity",
      "This cost function defines a residual vector as r = v - vref, with v and vref as the current and reference "
      "frame velocities, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameMotion, int>(
          bp::args("self", "state", "activation", "vref", "nu"),
          "Initialize the frame velocity cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param vref: reference frame velocity\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameMotion>(
          bp::args("self", "state", "activation", "vref"),
          "Initialize the frame velocity cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param vref: reference frame velocity"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameMotion, int>(
          bp::args("self", "state", "vref", "nu"),
          "Initialize the frame velocity cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param vref: reference frame velocity\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameMotion>(
          bp::args("self", "state", "vref"),
          "Initialize the frame velocity cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param vref: reference frame velocity"))
      .add_property("reference", &CostModelFrameVelocity::get_reference<FrameMotion>,
                    &CostModelFrameVelocity::set_reference<FrameMotion>, "reference frame velocity")
      .add_property("vref",
                    bp::make_function(&CostModelFrameVelocity::get_reference<FrameMotion>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFrameVelocity::set_reference<FrameMotion>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame velocity");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

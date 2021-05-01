///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameRotation() {  // TODO: Remove once the deprecated update call has been removed in a future
                                  // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelFrameRotation> >();

  bp::class_<CostModelFrameRotation, bp::bases<CostModelResidual> >(
      "CostModelFrameRotation",
      "This cost function defines a residual vector as r = R - Rref, with R and Rref as the current and reference "
      "frame rotations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameRotation,
               std::size_t>(bp::args("self", "state", "activation", "Rref", "nu"),
                            "Initialize the frame rotation cost model.\n\n"
                            ":param state: state of the multibody system\n"
                            ":param activation: activation model\n"
                            ":param Rref: reference frame rotation\n"
                            ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameRotation>(
          bp::args("self", "state", "activation", "Rref"),
          "Initialize the frame rotation cost model.\n\n"
          "The default nu value is obtained from model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Rref: reference frame rotation"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation, std::size_t>(
          bp::args("self", "state", "Rref", "nu"),
          "Initialize the frame rotation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameRotation>(
          bp::args("self", "state", "Rref"),
          "Initialize the frame rotation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Rref: reference frame rotation"))
      .add_property("reference", &CostModelFrameRotation::get_reference<FrameRotation>,
                    &CostModelFrameRotation::set_reference<FrameRotation>, "reference frame rotation")
      .add_property("Rref",
                    bp::make_function(&CostModelFrameRotation::get_reference<FrameRotation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFrameRotation::set_reference<FrameRotation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame rotation");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

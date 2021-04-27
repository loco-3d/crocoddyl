///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFrameTranslation() {  // TODO: Remove once the deprecated update call has been removed in a future
                                     // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelFrameTranslation> >();

  bp::class_<CostModelFrameTranslation, bp::bases<CostModelResidual> >(
      "CostModelFrameTranslation",
      "This cost function defines a residual vector as r = t - tref, with t and tref as the current and reference "
      "frame translations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameTranslation, int>(
          bp::args("self", "state", "activation", "xref", "nu"),
          "Initialize the frame translation cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameTranslation>(
          bp::args("self", "state", "activation", "xref"),
          "Initialize the frame translation cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference frame translation"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation, int>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the frame translation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation>(
          bp::args("self", "state", "xref"),
          "Initialize the frame translation cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation"))
      .add_property("reference", &CostModelFrameTranslation::get_reference<FrameTranslation>,
                    &CostModelFrameTranslation::set_reference<FrameTranslation>, "reference frame translation")
      .add_property("xref",
                    bp::make_function(&CostModelFrameTranslation::get_reference<FrameTranslation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFrameTranslation::set_reference<FrameTranslation>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame translation");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

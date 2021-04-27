///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-impulse.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactImpulse() {  // TODO: Remove once the deprecated update call has been removed in a future release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelContactImpulse> >();

  bp::class_<CostModelContactImpulse, bp::bases<CostModelResidual> >(
      "CostModelContactImpulse",
      "This cost function defines a residual vector as r = f-fref, where f,fref describe the current and reference "
      "the spatial impulses, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact impulse cost model.\n\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact impulse in the contact coordinates"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int>(
          bp::args("self", "state", "fref", "nr"),
          "Initialize the contact impulse cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact impulse in the contact coordinates\n"
          ":param nr: dimension of impulse vector (>= 6)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce>(
          bp::args("self", "state", "fref"),
          "Initialize the contact impulse cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact impulse in the contact coordinates"))
      .add_property("reference", &CostModelContactImpulse::get_reference<FrameForce>,
                    &CostModelContactImpulse::set_reference<FrameForce>,
                    "reference spatial contact impulse in the contact coordinates")
      .add_property("fref",
                    bp::make_function(&CostModelContactImpulse::get_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelContactImpulse::set_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference spatial contact impulse in the contact coordinates");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

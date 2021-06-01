///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseWrenchCone() {  // TODO: Remove once the deprecated update call has been removed in a future
                                      // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelImpulseWrenchCone> >();

  bp::class_<CostModelImpulseWrenchCone, bp::bases<CostModelResidual> >(
      "CostModelImpulseWrenchCone",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameWrenchCone>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the impulse Wrench cone cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame Wrench cone"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameWrenchCone>(
          bp::args("self", "state", "fref"),
          "Initialize the impulse force cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame Wrench cone"))
      .add_property("reference", &CostModelImpulseWrenchCone::get_reference<FrameWrenchCone>,
                    &CostModelImpulseWrenchCone::set_reference<FrameWrenchCone>, "reference frame Wrench cone");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

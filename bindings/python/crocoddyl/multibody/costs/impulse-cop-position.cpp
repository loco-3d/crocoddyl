///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Duisburg-Essen, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-cop-position.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseCoPPosition() {  // TODO: Remove once the deprecated update call has been removed in a future
                                       // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelImpulseCoPPosition> >();

  bp::class_<CostModelImpulseCoPPosition, bp::bases<CostModelResidual> >(
      "CostModelImpulseCoPPosition",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport>(
          bp::args("self", "state", "activation", "cop_support"),
          "Initialize the impulse CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
          ":param cop_support: impulse frame Id and cop support region"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport>(
          bp::args("self", "state", "cop_support"),
          "Initialize the impulse CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: impulse frame Id and cop support region"))
      .add_property("reference", &CostModelImpulseCoPPosition::get_reference<FrameCoPSupport>,
                    &CostModelImpulseCoPPosition::set_reference<FrameCoPSupport>, "reference foot geometry");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

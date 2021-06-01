///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseCoM() {  // TODO: Remove once the deprecated update call has been removed in a future
                               // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelImpulseCoM> >();

  bp::class_<CostModelImpulseCoM, bp::bases<CostModelResidual> >(
      "CostModelImpulseCoM",
      "This cost function defines a residual vector as r = Jcom * (vnext-v), with Jcom as the CoM Jacobian, and vnext "
      "the velocity after impact and v the velocity before impact, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system"));

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

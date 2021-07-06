///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactFrictionCone() {  // TODO: Remove once the deprecated update call has been removed in a future
                                        // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelContactFrictionCone> >();

  bp::class_<CostModelContactFrictionCone, bp::bases<CostModelResidual> >(
      "CostModelContactFrictionCone",
      "This cost function defines a residual vector as r = A*f, where A, f describe the linearized friction cone and "
      "the spatial force, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameFrictionCone,
               std::size_t>(bp::args("self", "state", "activation", "fref", "nu"),
                            "Initialize the contact friction cone cost model.\n\n"
                            ":param state: state of the multibody system\n"
                            ":param activation: activation model\n"
                            ":param fref: frame friction cone\n"
                            ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameFrictionCone>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact friction cone cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame friction cone"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameFrictionCone, std::size_t>(
          bp::args("self", "state", "fref", "nu"),
          "Initialize the contact friction cone cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame friction cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameFrictionCone>(
          bp::args("self", "state", "fref"),
          "Initialize the contact friction cone cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame friction cone"))
      .add_property("reference", &CostModelContactFrictionCone::get_reference<FrameFrictionCone>,
                    &CostModelContactFrictionCone::set_reference<FrameFrictionCone>, "reference frame friction cone");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

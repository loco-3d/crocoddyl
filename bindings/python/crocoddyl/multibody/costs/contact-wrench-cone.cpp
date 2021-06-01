///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactWrenchCone() {  // TODO: Remove once the deprecated update call has been removed in a future
                                      // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelContactWrenchCone> >();

  bp::class_<CostModelContactWrenchCone, bp::bases<CostModelResidual> >(
      "CostModelContactWrenchCone",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameWrenchCone, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
          "Initialize the contact wrench cone cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame wrench cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameWrenchCone>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame wrench cone"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameWrenchCone, int>(
          bp::args("self", "state", "fref", "nu"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame wrench cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameWrenchCone>(
          bp::args("self", "state", "fref"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame wrench cone"))
      .add_property("reference", &CostModelContactWrenchCone::get_reference<FrameWrenchCone>,
                    &CostModelContactWrenchCone::set_reference<FrameWrenchCone>, "reference frame wrench cone");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl

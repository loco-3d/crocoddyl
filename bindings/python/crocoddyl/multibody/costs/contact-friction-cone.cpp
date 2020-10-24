///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactFrictionCone() {
  bp::class_<CostModelContactFrictionCone, bp::bases<CostModelAbstract> >(
      "CostModelContactFrictionCone",
      "This cost function defines a residual vector as r = A*f, where A, f describe the linearized friction cone and "
      "the spatial force, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameFrictionCone, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
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
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameFrictionCone, int>(
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
      .def<void (CostModelContactFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactFrictionCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact friction cone cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelContactFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelContactFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactFrictionCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact friction cone cost.\n\n"
          "It assumes that that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelContactFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelContactFrictionCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact friction cone cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelContactFrictionCone::get_reference<FrameFrictionCone>,
                    &CostModelContactFrictionCone::set_reference<FrameFrictionCone>, "reference frame friction cone");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactFrictionCone> >();

  bp::class_<CostDataContactFrictionCone, bp::bases<CostDataAbstract> >(
      "CostDataContactFrictionCone", "Data for contact friction cone cost.\n\n",
      bp::init<CostModelContactFrictionCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact friction cone cost data.\n\n"
          ":param model: contact friction cone cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Ru", bp::make_getter(&CostDataContactFrictionCone::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)")
      .add_property(
          "contact",
          bp::make_getter(&CostDataContactFrictionCone::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataContactFrictionCone::contact), "contact data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl

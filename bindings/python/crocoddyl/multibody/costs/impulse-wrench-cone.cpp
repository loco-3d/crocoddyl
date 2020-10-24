///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseWrenchCone() {
  bp::class_<CostModelImpulseWrenchCone, bp::bases<CostModelAbstract> >(
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
      .def<void (CostModelImpulseWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelImpulseWrenchCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the impulse Wrench cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelImpulseWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelImpulseWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelImpulseWrenchCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the impulse Wrench cone cost.\n\n"
          "It assumes that that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelImpulseWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelImpulseWrenchCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse Wrench cone cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelImpulseWrenchCone::get_reference<FrameWrenchCone>,
                    &CostModelImpulseWrenchCone::set_reference<FrameWrenchCone>, "reference frame Wrench cone");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataImpulseWrenchCone> >();

  bp::class_<CostDataImpulseWrenchCone, bp::bases<CostDataAbstract> >(
      "CostDataImpulseWrenchCone", "Data for impulse Wrench cone cost.\n\n",
      bp::init<CostModelImpulseWrenchCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create impulse Wrench cone cost data.\n\n"
          ":param model: impulse Wrench cone cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulse",
          bp::make_getter(&CostDataImpulseWrenchCone::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataImpulseWrenchCone::impulse), "impulse data associated with the current cost")
      .add_property("Arr_Rx", bp::make_getter(&CostDataImpulseWrenchCone::Arr_Rx, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataImpulseWrenchCone::Arr_Rx),
                    "Intermediate product of Arr (2nd deriv of Activation) with Rx (deriv of residue)");
}

}  // namespace python
}  // namespace crocoddyl
